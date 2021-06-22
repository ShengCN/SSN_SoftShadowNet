import random
import time 
import numbergen as ng
import imagen as ig
import numpy as np
import cv2

def get_single_light(x, y, x_density=512, y_density=256, scale=3.0, size=0.1)
    gs = ig.Composite(operator=np.add,
                    generators=[ig.Gaussian(
                                size=size,
                                scale=scale*(ng.UniformRandom(seed=seed+i+5)+1e-3),
                                x=ng.UniformRandom(seed=seed+i+1)-0.5,
                                y=((1.0-ng.UniformRandom(seed=seed+i+2) * y_fact) - 0.5),
                                aspect_ratio=0.7,
                                orientation=np.pi*ng.UniformRandom(seed=seed+i+3),
                                ) for i in range(num)],
                        position=(0, 0), 
                        xdensity=512)
    
    ibl = gs()
    return ibl

img = get_single_light(0.5,0.5)
