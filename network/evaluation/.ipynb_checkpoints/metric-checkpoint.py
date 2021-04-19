import numpy as np
import math

def rmse(I1, I2):
    """ Compute root min square errors between I1 and I2
    """
    diff = I1 - I2
    num_pixels = float(diff.size)
    return np.sqrt(np.sum(np.square(diff))/ num_pixels)

def rmse_s(I1, I2):
    """
        compute loss and alpha for 
        
            min |a*I1 - I2|_2

        return alpha, scale invariant rmse
    """
    d1d1 = np.multiply(I1, I1)
    d1d2 = np.multiply(I1, I2)
    sum_d1d1, sum_d1d2 = np.sum(d1d1), np.sum(d1d2)
    
    if sum_d1d1 > 0.0:
        s = sum_d1d2/sum_d1d1
        return s, rmse(s * I1, I2)
    else:
        s = 1.0
        return s, rmse(I1, I2)

def norm_correlation(I1, I2):
    pass

if __name__ == '__main__':
    # prepare testing for the two functions
    h,w = 512, 512
    test_1,test_2 = np.zeros((h,w,3)), np.ones((h,w,3))

    print('rmse: {}'.format(rmse(test_1, test_2)))
    
    print('rmse_s: {}'.format(rmse_s(test_1, test_2)))


    # print(rmse(test_1, test_2))

    