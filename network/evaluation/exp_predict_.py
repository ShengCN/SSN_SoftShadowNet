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
import imageio

def get_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def get_folders(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]


def to_net_ibl(ibl_file):
    """ input:  32 x 16
        output: 32 x 5
    """
    def normalize_energy(ibl, energy=30.0):
        sum_ibl = np.sum(ibl)
        if sum_ibl < 1e-3:
            return ibl * 0.0
        return ibl * energy / sum_ibl

    ibl = imageio.imread(ibl_file)
    if np.uint8 == ibl.dtype:
       ibl = ibl / 255.0

    if len(ibl.shape) == 3:
        ibl = ibl[:5,:,0] + ibl[:5,:,1] + ibl[:5,:,2]
    else:
        ibl = ibl[:5, :]
    
    # return ibl
    return normalize_energy(ibl)

to_tensor = ToTensor()

def net_render(model, device, mask_file, ibl_file, out_file, save_npy=True):
    mask_np = imageio.imread(mask_file)
    mask_np = mask_np[:,:,0]
    if mask_np.dtype == np.uint8:
        mask_np = mask_np/255.0
    
    net_ibl = to_net_ibl(ibl_file)
    if net_ibl.shape[0] != 5 and net_ibl.shape[1] != 32:
        print('net render ibl is wrong, please check: ', net_ibl.shape)
        return None

    ibl = cv2.flip(net_ibl, 0)
    mask, ibl = to_tensor(np.expand_dims(mask_np, axis=2)), to_tensor(np.expand_dims(cv2.resize(ibl, (32,16)), axis=2))
    with torch.no_grad():
        I_s, L_t = torch.unsqueeze(mask.to(device),0), torch.unsqueeze(ibl.to(device),0)

        predicted_img, predicted_src_light = model(I_s, L_t)

    shadow_predict = np.squeeze(predicted_img[0].detach().cpu().numpy().transpose((1,2,0)))
    
    if save_npy:
        np.save(out_file, shadow_predict)
    
    dirname, fname = os.path.dirname(out_file), os.path.splitext(os.path.basename(out_file))[0]
    png_output = os.path.join(dirname, fname + '.png')
    cv2.normalize(shadow_predict, shadow_predict, 0.0, 1.0, cv2.NORM_MINMAX)
    plt.imsave(png_output, shadow_predict, cmap='gray')
    
    return shadow_predict

def predict(model, mts_gt_folder, out_folder, dev):
    def save_create_folder(dir, sub_folder):
        new_folder = join(dir, sub_folder)
        os.makedirs(new_folder, exist_ok=True)
        return new_folder

    def find_ibl_mask(files):
        ibl, mask = '', ''
        for f in files:
            if f.find('ibl.png') != -1:
                ibl = f

            if f.find('mask.png') != -1:
                mask = f
        
        return ibl, mask

    model_folders = get_folders(mts_gt_folder)
    ibl_num = len(get_folders(join(model_folders[0], 'pattern')))
    total = len(model_folders) * ibl_num

    with tqdm(total=total) as tbar:
        for model_folder in tqdm(model_folders):        
            model_name = os.path.basename(model_folder)
            
            exp_out_model = save_create_folder(out_folder, model_name)

            ibl_root_folder = join(model_folder, 'pattern')
            exp_ibl_out = save_create_folder(exp_out_model, 'pattern')

            ibl_outputs = get_folders(ibl_root_folder)
            for cur_out in ibl_outputs:
                ibl_file, mask_file = find_ibl_mask(get_files(cur_out))
                prefix = os.path.splitext(os.path.basename(mask_file))[0]
                
                cur_dir = os.path.basename(cur_out)
                exp_pred_out = save_create_folder(exp_ibl_out, cur_dir)

                net_render(model, dev, mask_file, ibl_file, join(exp_pred_out, '{}_predict.npy').format(prefix))
                tbar.update()
            

if __name__ == "__main__":    
    device = torch.device('cpu')
