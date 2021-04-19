import numpy as np
import numbergen as ng
import imagen as ig
import cv2
import random
from tqdm import tqdm

class base_ibl_animator(object):
    def __init__(self, num, size,verbose=True):
        random.seed(19920208)
        
        self.num = num
        self.verbose = verbose
        self.cur_pos = []
        self.cur_vec = []
        self.cur_size = np.array([size] * num) 
        self.cur_resize = np.array([random.random() for i in range(num)]) # < 0.5, smaller

        for i in range(num):
            self.cur_pos.append((512/2, 80/2))
            self.cur_vec.append((random.random() * 2.0 - 1.0, random.random() * 2.0 - 1.0))
        
        self.cur_pos = np.array(self.cur_pos)
        self.cur_vec = np.array(self.cur_vec)


        # normalize vector
        for i in range(num):
             self.cur_vec[i] =  self.cur_vec[i] / np.linalg.norm(self.cur_vec[i], 2)

        if self.verbose:
            print('pos: \n ', self.cur_pos)
            print('vec: \n ', self.cur_vec)
            print('size:\n ',self.cur_resize)

    def compute_center(self, j):
        center_vel = self.cur_vec[j]
        center_vel[:] = 0.0

        for i, p in enumerate(self.cur_vec):
            if i == j:
                continue
            
            center_vel += self.cur_vec[i] / len(self.cur_vec)
        
        return center_vel

    def move_advance(self):
        for i, p in enumerate(self.cur_pos):
            self.cur_pos[i] += self.cur_vec[i] * 2.0

            if self.num != 1:
                center_vel = self.compute_center(i)
            else:
                center_vel = -self.cur_vec[i]
            factor = 1.0
            self.cur_vec[i][0] = -1.0 * factor * center_vel[0] + (1.0-factor) * (random.random() * 2.0 - 1.0)
            self.cur_vec[i][1] = -1.0 * factor * center_vel[1] + (1.0-factor) * (random.random() * 2.0 - 1.0)
            
            speed = 0.4
            self.cur_vec[i] = self.cur_vec[i]/np.linalg.norm(self.cur_vec[i], 2) * speed

            if self.cur_pos[i][0] >= 511 or self.cur_pos[i][0]<=1:
                self.cur_vec[i][0] = -self.cur_vec[i][0]
            
            if self.cur_pos[i][1] >= 79 or self.cur_pos[i][1] <=1:
                self.cur_vec[i][1] = -self.cur_vec[i][1]


    def resize_advance(self):
        for i, p in enumerate(self.cur_size):
            if self.cur_size[i] < 0.002:
                self.cur_resize[i] = 0.7

            if self.cur_size[i] > 0.19:
                self.cur_resize[i] = 0.3

            if self.cur_resize[i] <= 0.5:
                self.cur_size[i] = self.cur_size[i] * 0.95
            else:
                self.cur_size[i] = self.cur_size[i] * 1.05
        
        self.cur_size = np.clip(self.cur_size, 0.001, 0.2) 

    # compute current ibl pattern given position and size list
    def get_cur_ibl(self):
        gs = ig.Composite(operator=np.add,
                generators=[ig.Gaussian(
                            size=self.cur_size[i],
                            scale=1.0,
                            x=(self.cur_pos[i][0]/512)-0.5,
                            y=(1.0 - self.cur_pos[i][1]/256)-0.5,
                            aspect_ratio=1.0,
                            ) for i in range(self.num)],
                    xdensity=512)
        return gs()

    def print_status(self):
        print('position: ')
        print(self.cur_pos)
        print('velocitiy: ')
        print(self.cur_vec)
        print('size: ')
        print(self.cur_size)

    # interface 
    def animate_ibl(self, iteration, max_iter):
        # 24 * 30 = 720 frames
        # 24 * 15 = 360, just move
        if iteration < max_iter//2:
            self.move_advance()
            # self.print_status()   

            return self.get_cur_ibl()

        # 24 * 5 = 360 + 120 = 480, scale the blob size
        if iteration >= max_iter//2 and iteration < max_iter//2 + max_iter//6: 
            self.resize_advance()
            return self.get_cur_ibl()

        self.move_advance()
        return self.get_cur_ibl()