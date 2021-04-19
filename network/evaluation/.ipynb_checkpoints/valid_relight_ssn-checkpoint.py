import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
from tqdm import tqdm
import numpy as np
import os
import math
from PIL import Image
from ssn.ssn_dataset import Mask_Transform, ToTensor, IBL_Transform
from ssn.ssn import Relight_SSN
from utils.net_utils import save_model, get_lr, set_lr
from utils.utils_file import create_folder
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
from params import params as options, parse_params
import multiprocessing
from multiprocessing import set_start_method
import random
from animation import *

# params = parse_params()

params = options().get_params()
print(params)

device = torch.device("cpu")
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = Relight_SSN(1,1)
weight_file = os.path.join('../weights', 'new_pattern_06-May-05-42-PM.pt')
# weight_file = os.path.join('weights', '1_ibl_14-April-10-34-PM.pt')

checkpoint = torch.load(weight_file, map_location=device)    
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])

def to_one_batch(img_tensor):
    c,h,w = img_tensor.size()
    return img_tensor.view(1,c,h,w)

def mask_to_rgb(mask_np):
    dimension = mask_np.shape
    w,h = dimension[0], dimension[1]
    rgb_np = np.zeros((w,h,3))
    rgb_np[:,:,0],rgb_np[:,:,1],rgb_np[:,:,2] = np.squeeze(mask_np),np.squeeze(mask_np),np.squeeze(mask_np)
    
    return rgb_np
    
def save_results(img_batch, out_path):
    # check if folder exist
    folder = os.path.dirname(out_path)
    create_folder(folder)

    batch, c, h, w = img_batch.size()
    for i in range(batch):
        cur_batch = img_batch[i]
        batch_np = cur_batch.detach().cpu().numpy().transpose((1,2,0))
        mask, shadow = batch_np[:,:,0], batch_np[:,:,1]
        # shadow[np.where(mask != 0)] = 0.0
        fig, axs = plt.subplots(1,2)
        for ax, cur_img, title in zip(axs, [mask, shadow], ['mask', 'shadow']):
            ax.imshow(cur_img, interpolation='nearest', cmap='gray')
            ax.set_title(title)
        plt.savefig(out_path)

def predict(img, ibl_img):
    """ Predict results for a numpy img(png image using alpha channel to represent mask) + ibl numpy img
        img: w x h x 3 image, [0,255]
        ibl_img: [0,1.0]
    """
    img_trnsf = transforms.Compose([
        Mask_Transform(),
        ToTensor()
    ])
    ibl_trnsf = transforms.Compose([
        # IBL_Transform(),
        ToTensor()
    ])

    ibl_tensor = ibl_trnsf(ibl_img)
    c,h,w =ibl_tensor.size()
    ibl_tensor = ibl_tensor.view(1, c, h, w)
    # print('ibl: ', ibl_tensor.size())

    img_tensor = img_trnsf(img)
    c,h,w = img_tensor.size()
    img_tensor = img_tensor.view(1, c, h, w)
    model.eval()
    with torch.no_grad():
        I_s = img_tensor.to(device)
        L_t = ibl_tensor.to(device)
        predicted_img = model(I_s, L_t)
        predicted_img = predicted_img[0].detach().cpu().numpy()
        predicted_img = predicted_img[0].transpose((1,2,0))
        return predicted_img

def compute_ibl(i,j, w=512, h=256):
    """ given width, height, (i,j) compute the 16x32 ibls """
    ibl = np.zeros((h,w,1))
    ibl[j,i] = 1.0
    ibl = gaussian_filter(ibl, 20)
    ibl = resize(ibl, (16,32))
    ibl = ibl/np.max(ibl)
    return np.squeeze(ibl)

def merge_result(pixel_img, mask_img, shadow_result):
    """ pixel image, mask image may in [0, 255]
    """
    if pixel_img.dtype == np.uint8:
        pixel_img = pixel_img/255.0
        
    if mask_img.dtype == np.uint8:
        mask_img = mask_img/255.0
    
#     print(pixel_img.shape)
#     print(mask_img.shape)
#     print(shadow_result.shape)
    
    h,w, c = shadow_result.shape
    merged_img = np.zeros((h,w,3))
    merged_img[:,:,0], merged_img[:,:,1], merged_img[:,:,2] = np.squeeze(shadow_result), np.squeeze(shadow_result), np.squeeze(shadow_result)
    merged_img[np.where(mask_img > 0.3)] = pixel_img[np.where(mask_img > 0.3)]
    return merged_img

def flipping_shadow(shadow_img, ibl_num):
    return ibl_num - shadow_img

def render_animation(target_img, target_mask_np, output_folder, ibl_animator, prefix_name=''):
    """ Given a mask(w x h x 3, uint8), render a sequence of images for making an animation"""
    
    def rotate_ibl(img_np, axis=1, step=1):
        """ rotate ibl along one axis for one pixel """
        new_np = np.copy(img_np)
        if axis == 1:
            # ---> direction
            new_np[:,step:] = img_np[:,:-step]
            new_np[:,0:step] = img_np[:,-step:]
        else:
            # | direction
            new_np[step:,:] = img_np[:-step,:]
            new_np[:step,:] = img_np[-step,:]

        return new_np
    
    def to_mask(target_mask_np):
        target_mask_np = (target_mask_np[:,:,0] + target_mask_np[:,:,1] + target_mask_np[:,:,2])/3.0
        target_mask_np = target_mask_np/np.max(target_mask_np)
        target_mask_np = resize(target_mask_np, (256,256,1))
        return target_mask_np
    
    def merge_save(target_img, target_mask_np, ibl, shadow_result,predict_fname):
        saving_result = merge_result(target_img, target_mask_np, shadow_result)
        saving_result[:16,:32] = 1.0 - ibl
        
        np.clip(saving_result, 0.0, 1.0, out=saving_result)
        plt.imsave(predict_fname, saving_result)

    def batch_predict(img, ibl_img):
        """ Predict results for a numpy img(png image using alpha channel to represent mask) + ibl numpy img
            img: batch x w x h x 3 image, [0,255]
            ibl_img: [0,1.0]
        """
        b, h, w, c = img.shape
        batch_img = torch.zeros([b,1,h,w])
        
        b, h, w, c = ibl_img.shape
        batch_ibl = torch.zeros([b,c,h,w])
        
        img_trnsf = transforms.Compose([
            Mask_Transform(),
            ToTensor()
        ])
        ibl_trnsf = transforms.Compose([
            # IBL_Transform(),
            ToTensor()
        ])

        for i in range(b):
            batch_img[i,:,:,:] = img_trnsf(img[i,:,:,:])
            batch_ibl[i,:,:,:] = ibl_trnsf(ibl_img[i,:,:,:])
            
        model.eval()
        with torch.no_grad():
            I_s = batch_img.to(device)
            L_t = batch_ibl.to(device)
            predicted_img,_ = model(I_s, L_t)
            predicted_img = predicted_img.detach().cpu().numpy()
            predicted_img = predicted_img.transpose((0, 2, 3, 1))
            
            # import pdb; pdb.set_trace()
            ibl_num = ibl_animator.get_ibl_num()
            for i in range(b):
                predicted_img[i] = flipping_shadow(predicted_img[i], ibl_num)
            
            return predicted_img
    
    
    ibl_num = ibl_animator.get_ibl_num()
    batch_size, batch_counter = 40, 0
    predict_fname_list = []

    # cur_ibl = get_first_ibl()
    prefix = 0
    batch_ibl = np.zeros((batch_size, 16, 32, 1))

    h,w,c = target_img.shape
    batch_mask_img = np.array([target_mask_np,] * batch_size)
                
    i_begin, i_end, i_step = 0, 512, 5
    j_begin, j_end, j_step = 150, 190, 10
    j_range = (j_end - j_begin) // j_step
    i_range = (i_end - i_begin) // i_step
    
    total = i_range * j_range
    for i in tqdm(range(total)):
        batch_counter += 1
        prefix += 1
        
        ibl = ibl_animator.animate_ibl(i, total)
        batch_ibl[batch_counter-1,:,:,:] = ibl
        
        out_fname = '{}_{:07d}.png'.format(prefix_name, prefix-1)
        predict_fname = os.path.join(output_folder, out_fname)
        predict_fname_list.append(predict_fname)
        
        if batch_counter == batch_size:
            
            # batch predict results
            batch_predict_result = batch_predict(batch_mask_img, batch_ibl)
            # save each batch results
            for bi in range(batch_size):
                predict_result = batch_predict_result[bi]
                # normalize to [0,1]
                predict_result = predict_result / ibl_num
                predict_result = np.clip(predict_result, 0.0, 1.0)
                merge_save(target_img, target_mask_np, batch_ibl[bi], predict_result, predict_fname_list[bi])
            batch_counter = 0
            predict_fname_list = []
                    
if __name__ == '__main__':
    pass
    # testing_ibl_file = os.path.join('/home/ysheng/Dataset/soft_shadow/single_human/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A','00000000_light.png')
    # testing_img = '/home/ysheng/Dataset/soft_shadow/real_human_testing_set/warrior-2-1.png'
    # img, ibl_img = plt.imread(testing_img), plt.imread(testing_ibl_file)
    # shadow_img = predict(img, ibl_img)
    # plt.imsave("testing.png",shadow_img)
    