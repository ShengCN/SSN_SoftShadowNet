import numpy as np
import math

def input_check(I1, I2):
    if len(I1.shape) != 2 or len(I2.shape) != 2:
        assert False, 'please check metric input'
        return False
    return True

def rmse(I1, I2):
    """ Compute root min square errors between I1 and I2
    """
    if not input_check(I1, I2):
        return float('nan')

    diff = I1 - I2
    num_pixels = float(diff.size)
    return np.sqrt(np.sum(np.square(diff))/ num_pixels)

def rmse_s(I1, I2):
    """
        compute loss and alpha for 
        
            min |a*I1 - I2|_2

        return alpha, scale invariant rmse
    """
    if not input_check(I1, I2):
        return 0.0, 0.0

    d1d1 = np.multiply(I1, I1)
    d1d2 = np.multiply(I1, I2)
    sum_d1d1, sum_d1d2 = np.sum(d1d1), np.sum(d1d2)
    
    if sum_d1d1 > 0.0:
        s = sum_d1d2/sum_d1d1
        return s, rmse(s * I1, I2)
    else:
        s = 1.0
        return s, rmse(I1, I2)

def ZNCC(I1, I2):
    """ 
        Zero-normalized cross-correlation (ZNCC)
        https://en.wikipedia.org/wiki/Cross-correlation
    """
    if not input_check(I1, I2):
        print('input shape is wrong')
        return 0.0

    diff = I1-I2
    num_pixels = float(diff.size)
    mu1, mu2 = np.sum(I1)/num_pixels, np.sum(I2)/num_pixels
    
    cen1, cen2 = I1 - mu1, I2 - mu2
    sig1, sig2 = np.sqrt(np.sum(np.multiply(cen1, cen1))/num_pixels), np.sqrt(np.sum(np.multiply(cen2, cen2))/num_pixels)
    if sig1 == 0 or sig2 == 0:
        return 0.0
    
    return np.sum(np.multiply(cen1, cen2))/(sig1 * sig2 * num_pixels)

if __name__ == '__main__':
    # prepare testing for the two functions
    h,w = 512, 512
    test_1, test_2 = np.zeros((h,w)), np.ones((h,w))
    test_random = np.random.rand(h,w)

    print('rmse: {}'.format(rmse(test_1, test_2)))
    print('rmse_s: {}'.format(rmse_s(test_1, test_2)))

    print('ZNCC: {}'.format(ZNCC(test_random, test_random)))
    print('ZNCC: {}'.format(ZNCC(test_random * 0.5, test_random)))
    # print(rmse(test_1, test_2))

    