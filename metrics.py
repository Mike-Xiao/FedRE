from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import numpy as np
import cv2
import lpips
import torch

# 初始化LPIPS
loss_fn_lpips = lpips.LPIPS(net='alex')

def calculate_metrics(img1, img2, mask=None):
    img1 = img1.float().numpy()
    img2 = img2.float().numpy()
    print(img1.shape,img2.shape)

    if mask is not None:
        mask = mask.float().numpy()
        # # Create a binary mask where the pixel value is True if the corresponding pixel in the mask is greater than 250
        # binary_mask = mask > 250
        # # Use the binary mask to select the pixels in img1 and img2
        # img1 = img1[binary_mask]
        # img2 = img2[binary_mask]
        img1 = img1 * (mask / 255)
        img2 = img2 * (mask / 255)

    print(img1.shape,img2.shape)
    
    # 计算SSIM
    ssim_value = ssim(img1, img2,data_range=img1.max() - img1.min(), channel_axis=0)

    # 计算MSE
    mse_value = mean_squared_error(img1, img2)

    # 计算PSNR
    psnr_value = cv2.PSNR(img1, img2)

    # 计算LPIPS
    lpips_value = loss_fn_lpips.forward(torch.from_numpy(img1), torch.from_numpy(img2))

    return ssim_value, mse_value, psnr_value, lpips_value

def get_high_intensity_pixels(image):
    # 获取所有像素值大于250的位置
    high_intensity_pixels = (image > 250)
    return high_intensity_pixels

def compute_masked_sensitivity(original_image, mask_image, compute_sens):
    # 获取掩码图片中像素值大于250的位置
    mask = get_high_intensity_pixels(mask_image)

    # 计算原始图片的每一层的敏感度
    layer_sensitivities = compute_sens(original_image)

    # 计算每一层在掩码区域的敏感度之和
    masked_sensitivity = []
    for layer_sensitivity in layer_sensitivities:
        masked_sensitivity.append(layer_sensitivity[mask].sum())

    return masked_sensitivity