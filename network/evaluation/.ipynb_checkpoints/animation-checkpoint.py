import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize

class base_ibl_animator(object):
    def __init__(self, ibl_num=1):
        self.ibl_num = ibl_num
        self.r = 200
        
    def get_ibl_num(self):
        return self.ibl_num
    
    def compute_ibl(self, i,j, w=512, h=256):
        """ given width, height, (i,j) compute the 16x32 ibls """
        ibl = np.zeros((h,w,1))
        
        i, j = i%(w-1), j%(h-1)
        ibl[j,i] = 1.0
        ibl = gaussian_filter(ibl, 20)
        ibl = resize(ibl, (16,32))
        ibl = ibl/np.max(ibl)
        return ibl
    
    # interface 
    def animate_ibl(self, iteration, max_iter):
        fract = iteration / max_iter
        i = int(512 * fract)
        return self.compute_ibl(i, self.r)

class three_ibl_animator(base_ibl_animator):
    def __init__(self, ibl_num=3):
        super().__init__(ibl_num)
        self.r = 150
        
    
    def animate_ibl(self, iteration, max_iter):
        # 256 x 512
        # rows: [150, 190, 2]
        rows = range(150, 190, 2)
        one_row_iterations = max_iter // (len(rows) -1)
        
        row_id = iteration // one_row_iterations
        col_fract = (iteration - row_id * one_row_iterations) / one_row_iterations
        col_id = int(col_fract * 512)
        
        middle_col = int(0.5 * 512)
        third_col = int((1.0-col_fract) * 512) - 1
        
        output_ibl = self.compute_ibl(col_id, rows[row_id])
        output_ibl += self.compute_ibl(middle_col,rows[row_id])
        output_ibl += self.compute_ibl(third_col,rows[row_id])
        
        return output_ibl
    
class bigger_gaussian_ibl_animator(base_ibl_animator):
    def __init__(self, ibl_num=1, size=2):
        super().__init__(ibl_num)
        self.row_begin = 150
        self.row_end = 190
        self.size = size
        
    def animate_ibl(self, iteration, max_iter):
        rows = range(self.row_begin, self.row_end, 2)
        one_row_iterations = max_iter // (len(rows) -1)
        
        row_id = iteration // one_row_iterations
        col_fract = (iteration - row_id * one_row_iterations) / one_row_iterations
        col_id = int(col_fract * 512)
        output_ibl = self.compute_ibl(col_id, rows[row_id])
        
        # make the light center bigger
        for i in range(self.size):
            for j in range(self.size):
                if i == 0 and j == 0:
                    continue
                output_ibl += self.compute_ibl(col_id + i, rows[row_id] + j)  
                
        return output_ibl
    
class random_ibl_animator(base_ibl_animator):
    def __init__(self, ibl_num=1):
        super().__init__(ibl_num)
        self.row_begin = 150
        self.row_end = 190
        
        np.random.seed(19920208)
        self.initial_position = np.random.randint(0, 512-1, size= self.ibl_num)
        self.random_direction = np.random.randint(0, 1, size=self.ibl_num) * 2 - 1
        
    def animate_ibl(self, iteration, max_iter):
        rows = range(self.row_begin, self.row_end, 2)
        one_row_iterations = max_iter // (len(rows) -1)
        
        row_id = iteration // one_row_iterations
        col_fract = (iteration - row_id * one_row_iterations) / one_row_iterations
        output_ibl = self.compute_ibl(512 // 2, rows[row_id])
        
        # random ibls
        for i in range(self.ibl_num):
            if i == 0:
                continue
            
            col_pos = self.initial_position[i] + self.random_direction[i] * 512 * col_fract
            output_ibl += self.compute_ibl(int(col_pos),rows[row_id])
            
        return output_ibl