


from sensitivity import compute_sens
import torch
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# 图片编号
image_number = "000000003"

# 使用字符串格式化设置图片路径
data_image_path = f"/home/tz_xiao/DocTamper_labeling/DocTamper_labeling/data/image/image-{image_number}.jpg"


label_image_path = f"/home/tz_xiao/DocTamper_labeling/DocTamper_labeling/data/label/image-{image_number}.jpg"
# 加载掩码图像
mask_image_path = f"/home/tz_xiao/DocTamper_labeling/DocTamper_labeling/data/privacy_mask/image-{image_number}.jpg"


def allocate_epsilon(epsilon_total, psi_scores, a=1):
    """
    根据PSI得分为每个层分配隐私预算ε，确保所有层的ε之和等于总隐私预算epsilon_total。

    参数:
    - epsilon_total: 总隐私预算。
    - psi_scores: 一个包含每个层级PSI得分的列表。
    - a: 调整系数，用于控制PSI得分对隐私需求的影响。

    返回:
    - epsilons: 一个列表，包含每个层级分配的隐私预算。
    """
    # 计算所有层级的PSI得分之和。
    total_psi = sum(psi_scores)
    # 计算每个层级的隐私需求。
    privacy_demands = [1 / (a * psi) for psi in psi_scores]
    # 计算每个层级的隐私预算。
    epsilons = [demand * epsilon_total / sum(privacy_demands) for demand in privacy_demands]
    return epsilons


def calculate_psi_scores(model,dataloader,device,loss_fn):

    mask_image = Image.open(mask_image_path).convert('L')

    # 应用阈值来创建二值图像
    threshold = 127
    mask_image = mask_image.point(lambda p: 255 if p > threshold else 0)

    # 将掩码图像转换回RGB
    mask_image = mask_image.convert('RGB')

    # 引入必要的库

    # 将PIL图像转换为numpy数组
    mask_np = np.array(mask_image)

    # 将numpy数组转换为PyTorch张量
    mask_tensor = torch.from_numpy(mask_np)
    mask = mask_tensor.permute(2, 0, 1)


    # Now you can pass the DataLoader to compute_sens
    sensitivity_list,vector_jacobian_products_cpu = compute_sens(model, dataloader, device,loss_fn)
    frobenius_norms = [torch.norm(tensor, dim=1) for tensor in vector_jacobian_products_cpu]
    
    mask_bi = mask > 128

    masked_sensitivity = [tensor[mask_bi[0][None, ...]].sum() for tensor in frobenius_norms]
    return masked_sensitivity