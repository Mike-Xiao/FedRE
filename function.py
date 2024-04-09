
import logging
import os
import time

import numpy as np
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from losses import soft_ce
from PSI import allocate_epsilon,calculate_psi_scores
from lib.utils.utils import AverageMeter
from lib.utils.utils import get_confusion_matrix
from lib.utils.utils import adjust_learning_rate
from lib.utils.utils import get_world_size, get_rank
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, final_output_dir):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    world_size = get_world_size()

    for i_iter, (images, labels, qtable) in enumerate(trainloader):
        # images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        losses, _ = model(images, labels, qtable)  # _ : output of the model (see utils.py)
        loss = losses.mean()

        reduced_loss = reduce_tensor(loss)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss)
            logging.info(msg)
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            global_steps += 1
            writer_dict['train_global_steps'] = global_steps

def train_nooptimizer(config, device, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, model, writer_dict, final_output_dir):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    world_size = get_world_size()

    grads = {}

    for i_iter, (images, labels, qtable) in enumerate(trainloader):
        # images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        losses, _ = model(images, labels, qtable)  # _ : output of the model (see utils.py)
        loss = losses.mean()

        reduced_loss = reduce_tensor(loss)

        model.zero_grad()
        loss.backward()

        # Accumulate gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name not in grads:
                    grads[name] = 0.
                grads[name] += param.grad.clone().detach()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        if i_iter % config.PRINT_FREQ == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  ' Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), print_loss)
            logging.info(msg)
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            global_steps += 1
            writer_dict['train_global_steps'] = global_steps
    # Average the gradients
    for name in grads:
        grads[name] /= len(trainloader)
        
        # 在train_nooptimizer函数中
    first_item = next(iter(grads.items()))
    print(f"{first_item[0]}: {first_item[1].shape}")
    # Return the gradients
    return grads



def train_withclip(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, final_output_dir, n_clients,  epsilons_total):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    world_size = get_world_size()
    
    # 设置裁剪值
    clip_norm = 2 # fedtamp2.0
    delta = 1/n_clients
    
    
    
    for i_iter, (images, labels, qtable) in enumerate(trainloader):
        images = images.cuda()
        labels = labels.long().cuda()

        loss_fn = soft_ce
        psi_scores = calculate_psi_scores(model,images,device,loss_fn)
        epsilons = allocate_epsilon(epsilons_total, psi_scores)
        
        losses, _ = model(images, labels, qtable)  # _ : output of the model (see utils.py)
        loss = losses.mean()

        reduced_loss = reduce_tensor(loss)

        model.zero_grad()
        loss.backward()

        # 计算梯度范数
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        # print(f"grad_norm: {grad_norm}")
        # 裁剪梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        
        # 加噪
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                noise_std = clip_norm  * math.sqrt(2 * num_epoch * math.log(1 / delta)) / epsilons[i]
                noise = torch.normal(mean=0, std=noise_std, size=param.grad.shape).to(param.grad.device)
                param.grad.add_(noise)
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                # 计算噪声标准差
                s_i = grad_norm / (epsilons[i] * math.sqrt(2 * T * math.log(1 / delta)))
                noise = torch.normal(mean=0, std=s_i, size=param.grad.shape).to(param.grad.device)
                param.grad.data.add_(noise) 
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss)
            logging.info(msg)
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            global_steps += 1
            writer_dict['train_global_steps'] = global_steps



def validate(config,device, testloader, model, writer_dict, valid_set="valid"):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    avg_mIoU = AverageMeter()
    avg_p_mIoU = AverageMeter()

    with torch.no_grad():
        for _, (image, label, qtable) in enumerate(tqdm(testloader)):
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)

            losses, pred = model(image, label, qtable)
            pred = F.upsample(input=pred, size=(
                        size[-2], size[-1]), mode='bilinear')
            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            current_confusion_matrix = get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            confusion_matrix += current_confusion_matrix
            # mIoU
            pos = current_confusion_matrix.sum(1)  # ground truth label count
            res = current_confusion_matrix.sum(0)  # prediction count
            tp = np.diag(current_confusion_matrix)  # Intersection part
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))  # Union part
            mean_IoU = IoU_array.mean()
            avg_mIoU.update(mean_IoU)
            TN = current_confusion_matrix[0, 0]
            FN = current_confusion_matrix[1, 0]
            FP = current_confusion_matrix[0, 1]
            TP = current_confusion_matrix[1, 1]
            p_mIoU = 0.5 * (FN / np.maximum(1.0, FN + TP + TN)) + 0.5 * (FP / np.maximum(1.0, FP + TP + TN))
            avg_p_mIoU.update(np.maximum(mean_IoU, p_mIoU))

    confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    
    
        # 在for循环结束后，从混淆矩阵中获取TP，FP和FN
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    # 计算precision和recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # 计算F1 Score
    F1 = 2 * (precision * recall) / (precision + recall)

    
    
    print_loss = ave_loss.average()/world_size

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar(valid_set+'_loss', print_loss, global_steps)
        writer.add_scalar(valid_set+'_mIoU', mean_IoU, global_steps)
        writer.add_scalar(valid_set+'_avg_mIoU', avg_mIoU.average(), global_steps)
        writer.add_scalar(valid_set+'_avg_p-mIoU', avg_p_mIoU.average(), global_steps)
        writer.add_scalar(valid_set+'_pixel_acc', pixel_acc, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, mean_IoU, avg_mIoU.average(), avg_p_mIoU.average(), IoU_array, pixel_acc, mean_acc, confusion_matrix, F1

