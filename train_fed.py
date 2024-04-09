import copy
import torch
from torch.utils.data import DataLoader, random_split
import sys, os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import pprint
import shutil

import logging
import time
import timeit
from pathlib import Path

import gc
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from lib import models
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train, validate, train_withclip
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank

from Splicing.data.data_core import SplicingDataset as splicing_dataset
from aggregate import aggregate_clients

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # args = parse_args()
    # Instead of using argparse, force these args:
    ## CHOOSE ##
    args = argparse.Namespace(cfg='../experiments/CAT_full.yaml', local_rank=0, opts=None)
    # args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', local_rank=0, opts=None)

    update_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    # build model
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config).to(device)
    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).to(device)
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    if config.DATASET.DATASET == 'splicing_dataset':
        ## CHOOSE ##
        train_dataset = splicing_dataset(crop_size=crop_size, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), mode='train', DCT_channels=1, read_from_jpeg=True, class_weight=[0.5, 2.5])  # full model
        # train_dataset = splicing_dataset(crop_size=crop_size, grid_crop=True, blocks=('DCTvol', 'qtable'), mode='train', DCT_channels=1, read_from_jpeg=True, class_weight=[0.5, 2.5])  # only DCT stream
        logger.info(train_dataset.get_info())
    else:
        raise ValueError("Not supported dataset type.")

    # 将数据集分成n个子集
    n_clients = 10
    epsilon_total = 50
    dataset_splits = random_split(train_dataset, [len(train_dataset) // n_clients for _ in range(n_clients)])

    # 为每个客户端创建一个DataLoader
    trainloaders = [DataLoader(dataset, 
                               batch_size=1, 
                               shuffle=config.TRAIN.SHUFFLE, 
                               num_workers=config.WORKERS, 
                               pin_memory=True) for dataset in dataset_splits]
    valid_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), mode="valid", DCT_channels=1, read_from_jpeg=True)  # full model
    # valid_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), mode="valid", DCT_channels=1, read_from_jpeg=True)  # only DCT stream
    validloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=20,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True )   
    # 创建n个模型副本和n个优化器
        # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights).to(device)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights).to(device)

    global_model = FullModel(model, criterion)
    
    # # 初始化全局模型
    # global_model = copy.deepcopy(model)
    
    # optimizer
    logger.info(f"# params with requires_grad = {len([c for c in model.parameters() if c.requires_grad])}, "
                f"# params freezed = {len([c for c in model.parameters() if not c.requires_grad])}")
    # 在每一轮训练中
    
    epoch_iters = int(train_dataset.__len__() /
                         config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    best_p_mIoU = 0
    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            best_p_mIoU = checkpoint['best_p_mIoU']
            last_epoch = checkpoint['epoch']
            model.model.module.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        else:
            logger.info("No previous checkpoint.")   
    
    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * epoch_iters

    # client_optimizer = torch.optim.SGD([{'params':
    #                             filter(lambda p: p.requires_grad,
    #                                     global_model.parameters()),
    #                         'lr': config.TRAIN.LR}],
    #                     lr=config.TRAIN.LR,
    #                     momentum=config.TRAIN.MOMENTUM,
    #                     weight_decay=config.TRAIN.WD,
    #                     nesterov=config.TRAIN.NESTEROV,
    #                     )

    # 在每一轮训练中
    for epoch in range(last_epoch, end_epoch):
        # 初始化全局参数
        # global_params = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}

        # total_samples = sum(len(loader.dataset) for loader in trainloaders)
        w_locals = []
        for client_id, client_dataloader in enumerate(trainloaders):
            line =  f"Training client {client_id + 1} of {len(trainloaders)} with {len(client_dataloader.dataset)} samples"
            print(line)
            logging.info(line)
            # 将全局模型的状态复制到客户端模型
            client_model = copy.deepcopy(global_model)
            
            
            # model_path = f"client_model_{client_id}.pth"
            # if os.path.exists(model_path):
            #     # 如果模型文件存在，就加载它
            #     client_model = torch.load(model_path)
            # else:
            #     # 否则，就复制全局模型
            #     client_model = copy.deepcopy(global_model)

            optimizer_path = f"optimizer_{client_id}.pth"
            if os.path.exists(optimizer_path):
                client_optimizer = torch.load(optimizer_path)
            else:
                client_optimizer = torch.optim.SGD([{'params':
                                          filter(lambda p: p.requires_grad,
                                                 client_model.parameters()),
                                                #  global_model.parameters()),
                                      'lr': config.TRAIN.LR}],
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
            epsilon_total = 50
            # 客户端训练模型
            train_withclip(config, epoch, config.TRAIN.END_EPOCH,
                    epoch_iters, config.TRAIN.LR, num_iters,
                    client_dataloader,client_optimizer, client_model, writer_dict, final_output_dir,n_clients,epsilon_total)

            # 保存客户端优化器
            torch.save(client_optimizer, optimizer_path)

            # # 保存更新后的模型和优化器
            # torch.save(client_model, f"client_model_{client_id}.pth")
        w_locals.append(copy.deepcopy(client_model.state_dict()))
        
        
        w_avg =  aggregate_clients(w_locals, psi_scores, global_model)
        global_model.load_state_dict(w_avg)
        
        
        del client_model               
        torch.cuda.empty_cache()
        gc.collect()

        # 验证
        if epoch % 1 == 0 or (epoch >= 80 and epoch % 5 == 0) or epoch >= 120:
            print("Start Validating..")
            writer_dict['valid_global_steps'] = epoch
            valid_loss, mean_IoU, avg_mIoU, avg_p_mIoU, IoU_array, pixel_acc, mean_acc, confusion_matrix, F1_score = \
                validate(config,device, validloader, global_model, writer_dict, "valid")

            torch.cuda.empty_cache()
            gc.collect()

            if avg_p_mIoU > best_p_mIoU:
                best_p_mIoU = avg_p_mIoU
                torch.save({
                    'epoch': epoch + 1,
                    'best_p_mIoU': best_p_mIoU,
                    'state_dict': global_model.model.module.state_dict(),
                    # 'optimizer': global_optimizer.state_dict(),
                }, os.path.join(final_output_dir, 'best.pth.tar'))
                logger.info("best.pth.tar updated.")

            msg = '(Valid) Loss: {:.3f}, MeanIU: {: 4.4f}, Best_p_mIoU: {: 4.4f}, avg_mIoU: {: 4.4f}, avg_p_mIoU: {: 4.4f}, Pixel_Acc: {: 4.4f}, Mean_Acc: {: 4.4f}, Mean_Acc: {: 4.4f}'.format(
                valid_loss, mean_IoU, best_p_mIoU, avg_mIoU, avg_p_mIoU, pixel_acc, mean_acc, F1_score)
            # msg = 'Epoch:{}, (Valid) Loss: {:.4f}, MeanIU: {: 8.8f}, Best_p_mIoU: {: 8.8f}, avg_mIoU: {: 8.8f}, avg_p_mIoU: {: 8.8f}, Pixel_Acc: {: 8.8f}, Mean_Acc: {: 8.8f}'.format(epoch,
            #     valid_loss, mean_IoU, best_p_mIoU, avg_mIoU, avg_p_mIoU, pixel_acc, mean_acc)
            logging.info(msg)
            logging.info(IoU_array)
            logging.info("confusion_matrix:")
            logging.info(confusion_matrix)
            logging.info("F1_score:")
            logging.info(F1_score)

        else:
            logging.info("Skip validation.")

        logger.info('=> saving checkpoint to {}'.format(
            os.path.join(final_output_dir, 'checkpoint.pth.tar')))
        # torch.save({
        #     'epoch': epoch + 1,
        #     'best_p_mIoU': best_p_mIoU,
        #     'state_dict': model.model.module.state_dict(),
        #     'optimizer': global_optimizer.state_dict(),
        # }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))
        # 在所有训练结束后
    # for client_id in range(len(trainloaders)):
    #     optimizer_path = f"optimizer_{client_id}.pth"
    #     if os.path.exists(optimizer_path):
    #         os.remove(optimizer_path)        
                
if __name__ == '__main__':
    main()   