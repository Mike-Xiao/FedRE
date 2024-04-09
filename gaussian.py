import torch
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0., std=1.):
    noise = torch.randn(image.size()) * std + mean
    noisy_image = image + noise
    # noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image

def visualize_image(image):
    # If the image has 3 channels, transpose it to be in the format (height, width, channels)
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    plt.imshow(image)
    plt.show()
