import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import cv2
import random
import imageio
from evaluation import evaluate, mitsuba_render,net_gt, net_render
from shutil import copyfile

def get_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def get_folders(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

def render_testing_set(dataset_folder):
    testing_ibl_folder = join(dataset_folder, 'ibls/pattern')
    test_model_folder= join(dataset_folder, 'test_models')
    result_folder = join(dataset_folder, 'evaluation')

    os.makedirs(result_folder, exist_ok=True)

    # prepare testing models
    model_lists = get_files(test_model_folder)
    test_model_list = []
    for m in model_lists:
        model_name = os.path.splitext(os.path.basename(m))[0]
        test_model_list.append(model_name)

    # test_model_list = random.sample(test_model_list, k=10, )

    # prepare mask diction
    # model -> masks
    mask_dict = dict()
    mask_root = join(dataset_folder, 'new_dataset/cache/test_mask')
    for model in test_model_list:
        mask_folder = join(mask_root, model)
        masks = get_files(mask_folder)
        filterd_masks = []
        for m in masks:
            if m.find('.png') != -1:
                filterd_masks.append(m)
        mask_dict[model] = filterd_masks

    ibl_files = get_files(testing_ibl_folder)
    print('model: {}, ibl: {}'.format(len(test_model_list), len(ibl_files)))

    # prepare mitsuba render scripts
    counter, total = 0, len(test_model_list) * len(ibl_files) 
    bash_file = 'mitsuba_bash.sh'
    if os.path.exists(bash_file):
        os.remove(bash_file)

    random.seed(19920208)
    with tqdm(total=total) as pbar:
        for model in test_model_list:
            output_folder = os.path.join(result_folder, model)
            cur_output = join(output_folder,'pattern')

            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(cur_output, exist_ok=True)

            for i, ibl_file in enumerate(ibl_files): 
                real_ibl, model_path = False, join(test_model_folder, model + ".obj")
                ibl_name = os.path.splitext(os.path.basename(ibl_file))[0]
                
                render_output = join(cur_output, ibl_name)
                os.makedirs(render_output, exist_ok=True)
                
                mask_path = random.sample(mask_dict[model], k=1)[0]
                # for mask_path in mask_list:  

                mask_name = os.path.splitext(os.path.basename(mask_path))[0]
                final_out_file, shadow_out_file = join(render_output, mask_name + '_mitsuba_final.exr'), join(render_output, mask_name + '_mitsuba_shadow.exr')
                mitsuba_render(mask_path, ibl_file, final_out_file, shadow_out_file, final = True, update_cam_param=True, real_ibl=real_ibl, write_cmd=True, skip=True)

                # copy mask and ibl to result folder
                copyfile(mask_path, join(render_output, mask_name + '.png'))
                copyfile(ibl_file, join(render_output, ibl_name + '.png'))

                with open(bash_file, 'a+') as f:
                    counter += 1
                    f.write('echo finish: {}\n'.format(counter/total))

                pbar.update()

if __name__ == '__main__':
    """ 
        input: dataset folder
        output: mitsuba rendered ground truth 

    """
    
    print('begin')
    render_testing_set('../dataset')