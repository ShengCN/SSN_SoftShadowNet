import sys
sys.path.append("..")

import os
from os.path import join
import torch
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
import random
import matplotlib.pyplot as plt
import cv2
from params import params
from .random_pattern import random_pattern
from .perturb_touch import random_perturb

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img, is_transpose=True):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if is_transpose:
            img = img.transpose((2, 0, 1))
        return torch.Tensor(img)

class SSN_Dataset(Dataset):
    def __init__(self, ds_dir, is_training):
        start = time.time()
        
        # # of samples in each group
        # magic number here
        self.ibl_group_size = 16
        
        parameter = params().get_params()

        # (shadow_path, mask_path)
        self.meta_data = self.init_meta(ds_dir)

        self.is_training = is_training
        self.to_tensor = ToTensor()
        
        end = time.time()
        print("Dataset initialize spent: {} ms".format(end - start))

        # fake random
        np.random.seed(19950220)
        np.random.shuffle(self.meta_data)
        
        self.valid_divide = 10
        if parameter.small_ds:
            self.meta_data = self.meta_data[:len(self.meta_data)//self.valid_divide]

        self.training_num = len(self.meta_data) - len(self.meta_data) // self.valid_divide
        print('training: {}, validation: {}'.format(self.training_num, len(self.meta_data) // self.valid_divide))
        
        self.random_pattern_generator = random_pattern()
        
        self.thread_id = os.getpid()
        self.seed = os.getpid()
        self.perturb = not parameter.pred_touch and not parameter.touch_loss
        
    def __len__(self):
        if self.is_training:
            return self.training_num
        else:
            # return len(self.meta_data) - self.training_num
            return len(self.meta_data) // self.valid_divide

    def __getitem__(self, idx):
        if self.is_training and idx > self.training_num:
            print("error")
        # offset to validation set
        if not self.is_training:
            idx = self.training_num + idx
        
        cur_seed = idx * 1234 + os.getpid() + time.time()
        random.seed(cur_seed)
        
        # random ibls
        shadow_path, mask_path, sketch_path, touch_path = self.meta_data[idx]  
        mask_img = plt.imread(mask_path)
        mask_img = mask_img[:,:,0]
        if mask_img.dtype == np.uint8:
            mask_img = mask_img/ 255.0

        mask_img, shadow_bases = np.expand_dims(mask_img, axis=2), 1.0 - np.load(shadow_path)
       
        shadow_img, light_img = self.render_new_shadow(shadow_bases, cur_seed)

        h,w = mask_img.shape[0], mask_img.shape[1] 
        touch_img = self.read_img(touch_path)
        touch_img = touch_img[:,:,0:1] 
        
#         if self.perturb:
#             touch_img = random_perturb(touch_img)
        
        input_img = np.concatenate((mask_img, touch_img), axis=2)
        input_img, shadow_img, light_img = self.to_tensor(input_img), self.to_tensor(shadow_img),self.to_tensor(light_img)
        return input_img, light_img, shadow_img
    
    def read_img(self, img_path):
        img = plt.imread(img_path)
        if img.dtype == np.uint8:
            img = img/ 255.0
        return img

    def init_meta(self, ds_dir):
        base_folder = join(ds_dir, 'base')
        mask_folder = join(ds_dir, 'mask')
        sketch_folder = join(ds_dir, 'sketch')
        touch_folder = join(ds_dir, 'touch')
        model_list = [f for f in os.listdir(base_folder) if os.path.isdir(join(base_folder, f))]
        metadata = []
        for m in model_list:
            shadow_folder, cur_mask_folder = join(base_folder, m), join(mask_folder, m)
            shadows = [f for f in os.listdir(shadow_folder) if f.find('_shadow.npy')!=-1]
            for s in shadows:
                prefix = s[:s.find('_shadow')]
                metadata.append((join(shadow_folder, s), 
                                join(cur_mask_folder, prefix + '_mask.png'), 
                                join(join(sketch_folder, m), prefix + '_sketch.png'),
                                join(join(touch_folder, m), prefix + '_touch.png')))
        
        return metadata

    def get_prefix(self, path):
        folder = os.path.dirname(path)
        basename = os.path.basename(path)
        return os.path.join(folder, basename[:basename.find('_')])
    
    def render_new_shadow(self, shadow_bases, seed):
        shadow_bases = shadow_bases[:,:,:,:]
        h, w, iw, ih = shadow_bases.shape

        num = random.randint(0, 50)
        pattern_img = self.random_pattern_generator.get_pattern(iw, ih, num=num, size=0.1, mitsuba=False, seed=int(seed))
        
        # flip to mitsuba ibl
        pattern_img = self.normalize_energy(cv2.flip(cv2.resize(pattern_img, (iw, ih)), 0))
        shadow = np.tensordot(shadow_bases, pattern_img, axes=([2,3], [1,0]))
        pattern_img = np.expand_dims(cv2.resize(pattern_img, (32,16)), 2)

        return np.expand_dims(shadow, 2), pattern_img
    
    def get_min_max(self, batch_data, name):
        print('{} min: {}, max: {}'.format(name, np.min(batch_data), np.max(batch_data)))

    def log(self, log_info):
        with open('log.txt', 'a+') as f:
            f.write(log_info)

    def normalize_energy(self, ibl, energy=30.0):
        if np.sum(ibl) < 1e-3:
            return ibl
        return ibl * energy / np.sum(ibl)
