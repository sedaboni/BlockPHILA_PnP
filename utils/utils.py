from scipy.interpolate import RegularGridInterpolator
from deepinv.optim.distance import Distance
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def tensor2array(img):
    img = img.cpu()
    img = img.squeeze(0).detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

def save_image(img, filename, folder, flag='x'):
    img = tensor2array(img)
    img = np.clip(img, 0, 1)

    folder_img = f'{folder}/img'
    if not os.path.exists(folder_img):
        os.makedirs(folder_img)

    filepath = f'{folder_img}/{filename}_{flag}.png'
    plt.imsave(filepath, img)

def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf - 1) * 0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    if x.ndim == 2:
        interp = RegularGridInterpolator((xv, yv), x, method='linear')
        x = interp(np.meshgrid(x1, y1))
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            interp = RegularGridInterpolator((xv, yv), x[:, :, i], method='linear')
            x[:, :, i] = interp(np.meshgrid(x1, y1))
    return x