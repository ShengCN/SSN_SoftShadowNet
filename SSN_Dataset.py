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
import h5py

from random_pattern import random_pattern


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
    def __init__(self, opt: dict, is_training: bool):
        """ Inputs:
               opt: {'hdf5_file': ...,
                     'use_ao': True/False    # do we use AO?
                     'use_64_16': True/False # do we use 512 x 512 x 64 x 16 IBL?
        }
        """

        self.hdf5_file   = opt['hdf5_file']
        self.use_ao      = opt['use_ao']
        self.use_64_16   = opt['use_64_16']
        self.is_training = is_training
        self.to_tensor   = ToTensor()

        self.init_meta(self.hdf5_file)
        self.random_pattern_generator = random_pattern()


    def __len__(self):
        if self.is_training:
            return self.train_n
        else:
            return self.valid_n


    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.training_num + idx

        cur_scene = self.meta_data[idx]

        x    = self.x[cur_scene][...]
        base = self.base[cur_scene]

        if not self.use_ao:
            x = x[..., 0:1]

        shadow_img, light_img = self.render_new_shadow(base)
        input_img, shadow_img, light_img = self.to_tensor(x), self.to_tensor(shadow_img), self.to_tensor(light_img)
        return {'x': input_img, 'ibl':light_img, 'y': shadow_img}


    def init_meta(self, hdf5_path: str):
        f = h5py.File(hdf5_path, 'r')

        self.x = f['x']

        if self.use_64_16:
            self.base_key = 'base_64_16'
            self.base     = f[self.base_key]
        else:
            self.base_key = 'base_32_8'
            self.base     = f[self.base_key]

        self.meta_data = [k for k in self.x.keys()]

        self.train_n = len(self.meta_data) - len(self.meta_data) // 10
        self.valid_n = len(self.meta_data) - self.train_n

        print('training: {}, validation: {}'.format(self.train_n, self.valid_n))


    def render_new_shadow(self, shadow_bases):
        shadow_bases = shadow_bases[:,:,:,:]
        h, w, iw, ih = shadow_bases.shape

        num = random.randint(0, 50)
        pattern_img = self.random_pattern_generator.get_pattern(iw, ih, num=num, size=0.1, mitsuba=False, dataset=True)
        shadow      = np.tensordot(shadow_bases, pattern_img, axes=([2,3], [1,0]))
        pattern_img = np.expand_dims(cv2.resize(pattern_img, (32,16)), 2)

        return np.expand_dims(shadow, 2), pattern_img


    def normalize_energy(self, ibl, energy=30.0):
        if np.sum(ibl) < 1e-3:
            return ibl
        return ibl * energy / np.sum(ibl)


if __name__ == '__main__':
    from tqdm import tqdm
    opt = {'hdf5_file': 'Dataset/human_data/all_base.hdf5', 'use_ao': False, 'use_64_16': False}

    ds = SSN_Dataset(opt, True)

    for i, d in enumerate(tqdm(ds, total=len(ds), desc='Test')):
        x = d['x']
        y = d['y']
        ibl = d['ibl']
