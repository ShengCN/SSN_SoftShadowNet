import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import cv2
import imageio
import metric

def get_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def get_folders(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

def get_fname(fpath):
    return os.path.splitext(os.path.basename(fpath))[0]


if __name__ == '__main__':
    final_us_folder = 'user_study_result'
    final_files = get_files(final_us_folder)
    print('there are {} files'.format(len(final_files)))
    prefix_set = set()
    for f in final_files:
        fname = os.path.splitext(os.path.basename(f))[0]
        if fname.find('predict') != -1:
            prefix = fname[:fname.find('_mask')]
            prefix_set.add(prefix)
        
    print('samples: {}'.format(len(prefix_set)))

    # prepare samples
    out_folder = 'final_us'
    os.makedirs(out_folder, exist_ok=True)

    for fs in final_samples:
        final, mts_shadow, predict = fs + '_mask_final.exr', fs + '_mask_shadow.exr', fs + '_mask_predict.npy'
        final, mts_shadow, predict = join(final_us_folder, final), join(final_us_folder, mts_shadow), join(final_us_folder, predict) 

        final_np, mts_shadow_np, predict_np = imageio.imread(final), imageio.imread(mts_shadow), np.load(predict)
        show(final_np)
        show(mts_shadow_np)
        show(pred_np)

        break