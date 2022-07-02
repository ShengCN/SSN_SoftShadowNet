import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2

from . import pytorch_ssim
from .vgg19_loss import VGG19Loss
from abc import ABC, abstractmethod
from collections import OrderedDict

class abs_loss(ABC):
    def loss(self, gt_img, pred_img):
        pass


class norm_loss(abs_loss):
    def __init__(self, norm=1):
        self.norm = norm


    def loss(self, gt_img, pred_img):
        """ M * (I-I') """
        b, c, h, w = gt_img.shape
        return torch.norm(gt_img-pred_img, self.norm)/(h * w * b)



class ssim_loss(abs_loss):
    def __init__(self, window_size=11):
        self.ssim_loss_ = pytorch_ssim.SSIM(window_size=window_size)


    def loss(self, gt_img, pred_img):
        b, c, h, w = gt_img.shape
        l = 1.0 - self.ssim_loss_(gt_img, pred_img)
        return l/b


class hierarchical_ssim_loss(abs_loss):
    def __init__(self, patch_list: list):
        self.ssim_loss_list = [pytorch_ssim.SSIM(window_size=ws) for ws in patch_list]


    def loss(self, gt_img, pred_img):
        b, c, h, w = gt_img.shape
        total_loss = 0.0
        for loss_func in self.ssim_loss_list:
            total_loss +=  (1.0-loss_func(gt_img, pred_img))

        return total_loss/b


class vgg_loss(abs_loss):
    def __init__(self):
        self.vgg19_ = VGG19Loss()


    def loss(self, gt_img, pred_img):
        b, c, h, w = gt_img.shape
        v = self.vgg19_(gt_img, pred_img, pred_img.device)
        return v/b



class grad_loss(abs_loss):
    def __init__(self, k=4):
        self.k = 4

    def loss(self, gt_img, pred_img):
        b, c, h, w = gt_img.shape


        grad_loss = 0.0
        diff      = gt_img - pred_img

        for i in range(self.k):
            div_factor               = 2 ** i
            cur_transform            = T.Resize([h // div_factor, ])
            cur_diff                 = cur_transform(diff)
            cur_diff_dx, cur_diff_dy = self.img_grad(cur_diff)

            h, w = cur_diff.shape[2:]
            grad_loss += (torch.sum(torch.abs(cur_diff_dx)) + torch.sum(torch.abs(cur_diff_dy))) / (h * w)

        return grad_loss/b


    def img_grad(self, img):
        """ Comptue image gradient by sobel filtering
            img: B x C x H x W
        """

        b, c, h, w = img.shape
        ysobel     = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        xsobel     = ysobel.transpose(0,1)

        xsobel_kernel = xsobel.float().unsqueeze(0).expand(1, c, 3, 3).to(img)
        ysobel_kernel = ysobel.float().unsqueeze(0).expand(1, c, 3, 3).to(img)

        dx = F.conv2d(img, xsobel_kernel, padding=1, stride=1)
        dy = F.conv2d(img, ysobel_kernel, padding=1, stride=1)
        return dx, dy



# def all_loss(gt_img, pred_img, loss_weights):
#     l1_weight   = loss_weights['L1']
#     SSIM_weight = loss_weights['SSIM']
#     VGG_weight  = loss_weights['VGG']
#     grad_weight = loss_weights['Grad']

#     b      = gt_img.shape[0]
#     nl     = avg_norm_loss(gt_img, pred_img, norm=1)/b
#     ssim_l = ssim_loss(gt_img, pred_img)/b
#     vgg_l  = vgg_loss(gt_img, pred_img)/b
#     grad_l = grad_loss(gt_img, pred_img)/b

#     vis = {'L1': nl, 'SSIM': ssim_l, 'VGG': vgg_l, 'Grad': grad_l}

#     sum_loss = l1_weight * nl + SSIM_weight * ssim_l + VGG_weight * vgg_l + grad_weight * grad_l
#     return sum_loss, vis


if __name__ == '__main__':
    a = torch.rand(3,3,128,128)
    b = torch.rand(3,3,128,128)

    import pdb; pdb.set_trace()
    print(grad_loss(a, b))
