import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import numpy as np
import os
from os.path import join
import datetime
import matplotlib.pyplot as plt
import cv2 
from ssn.ssn_dataset import ToTensor

# globals
tensor_convert = ToTensor()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_mask_ibl(path, general=True):
    """ Given gt path, return mask and ibl map
    """
    mask_root = '/home/ysheng/Dataset/benchmark_ds/mask'
    ibl_root = '/home/ysheng/Dataset/benchmark_ds/lights'
    if general:
        mask_root = join(mask_root, 'general')
    else:
        mask_root = join(mask_root, 'human')

    model_name = os.path.basename(os.path.dirname(path))
    basename = os.path.basename(path)
    pitch_rot = basename[basename.find('pitch'):basename.find('_mts')]
    ibl = basename[basename.find('mts')+len('mts_'):basename.find('.png')]
    
    mask_path = join(mask_root, join(model_name, pitch_rot + '_mask.png'))
    ibl_path = join(ibl_root, ibl + '.png')
    
    return mask_path, ibl_path

def to_net_ibl(ibl_file):
    """ input:  32 x 16
        output: 32 x 8
    """
    def normalize_energy(ibl, energy=30.0):
        sum_ibl = np.sum(ibl)
        if sum_ibl < 1e-3:
            return ibl * 0.0
        return ibl * energy / sum_ibl

    ibl = plt.imread(ibl_file)
    if np.uint8 == ibl.dtype:
        ibl = ibl / 255.0
    
    ibl = ibl[:80,:,:1]
    ibl = cv2.resize(ibl, (32,16))
    ibl = cv2.flip(ibl, 0)
    return normalize_energy(ibl)

def to_tensor(mask_np, ibl_np):
    assert len(mask_np.shape) == 2
    
    mask = mask_np[:,:,np.newaxis]
    ibl_np = ibl_np[:,:,np.newaxis]
    mask = tensor_convert(mask)
    ibl = tensor_convert(ibl_np)
    return mask.unsqueeze(0), ibl.unsqueeze(0)

def ssn_touch_pred(model, mask, ibl, dev, baseline=False, old_ssn=False):
    ibl_img = to_net_ibl(ibl)
    mask_img = plt.imread(mask)
    if mask_img.dtype == np.uint8:
        mask_img = mask_img/255.0
    
    mask_img = (mask_img[:,:,0] + mask_img[:,:,1] + mask_img[:,:,2])/3.0
    mask, ibl = to_tensor(mask_img, ibl_img)
    mask, ibl = mask.to(dev), ibl.to(dev)
    
    touch = torch.zeros((1,1,256,256)).to(dev)
    if not old_ssn:
        I_s = torch.cat((mask, touch), 1)
    else:
        I_s = mask
    
    I_s = I_s.to(dev)
    shadow, touch_pred = model(I_s, ibl)
    
    if not baseline:
        I_s = torch.cat((mask, touch_pred), 1)
        I_s = I_s.to(dev)
        shadow, touch_pred = model(I_s, ibl)
    
    return shadow[0].detach().cpu().numpy().transpose((1,2,0))

if __name__ == "__main__":    
    model = SSN_Touch()    
    model.to(device) 
    
    test_gt = '/home/ysheng/Dataset/benchmark_ds/shadow_gt/general/airplane_0685/shadow_pitch_15_rot_-28_fov_92_mts_1_0.png'
    mask, ibl = get_mask_ibl(test_gt)

    img = ssn_touch_pred(model, mask, ibl, device)
    print(img.shape)