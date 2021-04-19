import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
import os
from os.path import join
from evaluation import evaluate, mitsuba_render,net_gt, net_render
import shutil
import imageio 
import cv2
import metric

def get_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def get_folders(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

def make_animation(mask_file, out_fname, render_files, ibl_files, is_both=False, is_mts=False):
    ims = []
    fig = plt.figure()
    render_files.sort()
    ibl_files.sort()
    
    file_dname = os.path.dirname(render_files[0])
    final_filename = os.path.join(file_dname, '000000_final.exr')
    final_img = imageio.imread(final_filename)
    final_img = final_img/np.max(final_img)
    mask_np = plt.imread(mask_file)
    mask_np = mask_np[:,:,:3]

    for i, f in enumerate(tqdm(render_files)):
        plt.axis('off')
        if is_both:
            prefix = f[:f.find('_net.png')]
            mts_file = prefix + '_shadow.exr'
            mts_img = imageio.imread(mts_file)
            img = 1.0 - plt.imread(f)[:,:,:3]
            s,_ = metric.rmse_s(img[:,:,0], mts_img[:,:,0])
            img = img * s

            # composite final human
            mts_img[np.where(mask_np!=0)] = final_img[np.where(mask_np!=0)]
            img[np.where(mask_np!=0)] = final_img[np.where(mask_np!=0)]

            out_img = np.zeros((256, 256 * 2, 3))
            out_img[:,:256] = mts_img
            out_img[:,256:] = img

        else:
            if is_mts:
                out_img = imageio.imread(f)
            else:
                prefix = f[:f.find('_net.png')]
                mts_file = prefix + '_shadow.exr'
                mts_img = imageio.imread(mts_file)
                img = 1.0 - plt.imread(f)[:,:,:3]
                s,_ = metric.rmse_s(img[:,:,0], mts_img[:,:,0])
                out_img = img * s
            out_img[:,:] = final_img[np.where(mask_np!=0)]
            
        # cv2.normalize(out_img, out_img, 0.0, 1.0, cv2.NORM_MINMAX)
        ibl = plt.imread(ibl_files[i])
        ibl = cv2.resize(np.clip(cv2.resize(ibl, (32, 16)), 0.0, 1.0), (128,64))
        cv2.normalize(ibl, ibl, 0.0, 1.0, cv2.NORM_MINMAX)
        c = out_img.shape[2]

        h,w,_ = ibl.shape
        if is_both:
            offset = 512//2 - w//2
        else:
            offset = 0
        out_img[:h,offset:offset+w] = ibl[:,:, :c]
        im = plt.imshow(out_img, animated=True)
        ims.append([im])
        
    ani = animation.ArtistAnimation(fig, ims, interval=24, blit=True)
    ani.save(out_fname)

def render_animations(mask_file, render_out_folder, ani_out_folder, typename='mitsuba'):
    search_str = {'mitsuba':'shadow.exr', 'net':'net.png'}
    models = get_folders(render_out_folder)
    ori_ibl_folder = 'animation/'
    
    with tqdm(total=len(models)) as pbar:
        for model in models:
            ibl_folds = get_folders(model)
            print('{}: {}'.format(model, len(ibl_folds)))

            for ibl in tqdm(ibl_folds):
                model_name, ibl_name = os.path.basename(model), os.path.basename(ibl)
                
                ibl_files = get_files(join(ori_ibl_folder, ibl_name))
                mts_result_files = [join(ibl, f) for f in os.listdir(ibl) if (f.find(search_str['mitsuba']) != -1)]
                net_result_files = [join(ibl, f) for f in os.listdir(ibl) if (f.find(search_str['net']) != -1)]
                mts_out_fname = '{}_{}_mitsuab.mp4'.format(model_name, ibl_name)
                net_out_fname = '{}_{}_net.mp4'.format(model_name, ibl_name)
                
                mts_out_fname = join(ani_out_folder, mts_out_fname)
                net_out_fname = join(ani_out_folder, net_out_fname)
                
                if len(net_result_files) == 0:
                        continue

                if typename == 'mitsuba':
                    make_animation(mask_file, mts_out_fname, mts_result_files, ibl_files, is_mts=True)
                
                if typename == 'net':
                    make_animation(mask_file, net_out_fname, net_result_files, ibl_files)
                
                if typename == 'both':
                    # make_animation(net_out_fname, net_result_files, ibl_files)
                    # make_animation(mts_out_fname, mts_result_files, ibl_files)
                    mts_out_fname= '{}_{}_both.mp4'.format(model_name, ibl_name)
                    mts_out_fname = join(ani_out_folder,mts_out_fname)
                    make_animation(mask_file, mts_out_fname, net_result_files, ibl_files, is_both=True)
                
            pbar.update()
    
    print('render finished')

if __name__ == '__main__':
    rendering_out_folder = '/home/ysheng/Dataset/eval_animation/'
    animation_out = 'ani_out'; os.makedirs(animation_out, exist_ok=True)
    mask_file = '/home/ysheng/Dataset/new_dataset/cache/test_mask/simulated_combine_male_genesis8_matias_hywavybob_dsoset_Base_Pose_Walking_B/pitch_30_rot_60_mask.png'

    # render_animations(rendering_out_folder, animation_out, typename='mitsuba')
    render_animations(mask_file, rendering_out_folder, animation_out, typename='both')