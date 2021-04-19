import sys
sys.path.append("..")

import os 
from os.path import join
import argparse
import time
from tqdm import tqdm
from ssn.ssn import Relight_SSN
from ssn.ssn_dataset import ToTensor
import pickle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import subprocess
import imageio
import shutil 

parser = argparse.ArgumentParser(description='evaluatoin pipeline')
parser.add_argument('-f', '--file', type=str, help='input model file')
parser.add_argument('-m', '--mask', type=str, help='mask file')
parser.add_argument('-i', '--ibl', type=str, help='ibl file')
parser.add_argument('-o', '--output', type=str, help='output folder')
parser.add_argument('-w', '--weight', type=str, help='weight of current model', default='../weights/group_norm_15-May-07-45-PM.pt')
parser.add_argument('-v', '--verbose', action='store_true', help='output file name')

options = parser.parse_args()
print('options: ', options)

# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = Relight_SSN(1,1)
weight_file = options.weight
checkpoint = torch.load(weight_file, map_location=device)    
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])

to_tensor = ToTensor()

cam_world_dict = {}
def get_files(folder):
    return [join(folder, f) for f in os.listdir(folder) if os.path.isfile(join(folder, f))]

def parse_cam_world_str(lines):
    cam, world = [], ''
    cam = [lines[0], lines[1], lines[2]]
    world_str = lines[3] + ',' + lines[4] + ',' + lines[5] + ',' + lines[6]
    world_elements = world_str.split(',')

    world = ''
    for w in world_elements:
        world += w + ' '
    return cam, world

def parse_camera_world(update=False):
    cam_world_file = './cam_world_dict.pkl'
    cam_world_dict = dict()
    if not update and os.path.exists(cam_world_file):
        with open(cam_world_file, 'rb') as f:
            cam_world_dict = pickle.load(f)
    else:
        cam_world_folder = '/home/ysheng/Dataset/mts_params/'
        folders = [join(cam_world_folder, f) for f in os.listdir(cam_world_folder) if os.path.isdir(join(cam_world_folder, f))]
        # print('there are {} folders'.format(len(folders)))
        for f in folders:
            basename = os.path.basename(f)
            if basename not in cam_world_dict.keys():
                cam_world_dict[basename] = dict()

            cam_world_files = get_files(f)
            for cam_world in cam_world_files:
                if cam_world.find('txt') == -1:
                    continue

                lines = []
                with open(cam_world) as f:
                    for l in f:
                        lines.append(l.rstrip('\n'))

                fname = os.path.splitext(os.path.basename(cam_world))[0]
                cam_world_dict[basename][fname] = parse_cam_world_str(lines)
        
        with open(cam_world_file, 'wb') as handle:
            pickle.dump(cam_world_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cam_world_dict

def composite_shadow(mask_np, mitsuba_final_np, mitsuba_shadow_np):
    masked_area = np.where(mask_np > 1e-3)
    
    ret = np.copy(mitsuba_shadow_np)
    ret[masked_area] = mitsuba_final_np[masked_area]
    return ret

def to_net_ibl(ibl_file):
    """ input:  32 x 16
        output: 32 x 5
    """
    def normalize_energy(ibl, energy=30):
        sum_ibl = np.sum(ibl)
        if sum_ibl < 1e-3:
            return ibl * 0.0
        return ibl * energy / sum_ibl

    ibl = imageio.imread(ibl_file)
    if np.uint8 == ibl.dtype:
        ibl = ibl / 255.0

    if len(ibl.shape) == 3:
        ibl = ibl[:5,:,0] + ibl[:5,:,1] + ibl[:5,:,2]
    else:
        ibl = ibl[:5, :]
    
    # return ibl
    return normalize_energy(ibl)

model_root = '/home/ysheng/Dataset/models'
mitsuba_bash = '/home/ysheng/Documents/mitsuba/dist/mitsuba'
mts_final_xml,mts_shadow_xml = '/home/ysheng/Documents/adobe_shadow_net/evaluation/mts_final.xml', '/home/ysheng/Documents/adobe_shadow_net/evaluation/mts_shadow.xml'
def mitsuba_render(mask_file, ibl_file, final_out_file, shadow_out_file, final=True, update_cam_param=False, real_ibl=True, write_cmd=False, skip=True, cmd_path='mitsuba_bash.sh'):
    """ Input: mitsuba rendering related resources
        Output: rendered_gt, saved shadow image 
    """
    final_out_folder, shadow_out_folder = os.path.dirname(final_out_file), os.path.dirname(shadow_out_file)
    cam_world_dict = parse_camera_world(update_cam_param)

    # parse camera parameters, human matrix
    # model_name = os.path.splitext(os.path.basename(model_file))[0]
    # model_cam_world_dict = cam_world_dict.get(model_name)
    # cam, world = model_cam_world_dict[list(model_cam_world_dict.keys())[0]]
    model_name, mask_name = os.path.basename(os.path.dirname(mask_file)), os.path.splitext(os.path.basename(mask_file))[0]
    cam, world = cam_world_dict[model_name][mask_name]
    model_file = join(model_root, model_name + '.obj')

    # ground plane model path
    ground_path = '"/home/ysheng/Dataset/ori_models/ground/ground.obj'

    samples = 256
    # prepare an xml into this folder that has parameter for model file and ibl file, output_file

    if skip:
        shadow_cmd = '\"{}\" {} -Dw=256 -Dh=256 -Dsamples={} -q -Dori=\"{}\" -Dtarget=\"{}\" -Dup=\"{}\" -Dibl=\"{}\" -Dground={}\" -Dmodel=\"{}\" -Dworld=\"{}\" -x -o \"{}\"'.format(
        mitsuba_bash, mts_shadow_xml, samples, cam[0], cam[1], cam[2], ibl_file, ground_path, model_file, world, shadow_out_file)

        final_cmd = '\"{}\" {} -Dw=256 -Dh=256 -Dsamples={} -q -Dori=\"{}\" -Dtarget=\"{}\" -Dup=\"{}\" -Dibl=\"{}\" -Dground={}\" -Dmodel=\"{}\" -Dworld=\"{}\" -x -o \"{}\"'.format(
        mitsuba_bash, mts_final_xml, samples, cam[0], cam[1], cam[2], ibl_file, ground_path, model_file, world, final_out_file)
    else:
        shadow_cmd = '\"{}\" {} -Dw=256 -Dh=256 -Dsamples={} -q -Dori=\"{}\" -Dtarget=\"{}\" -Dup=\"{}\" -Dibl=\"{}\" -Dground={}\" -Dmodel=\"{}\" -Dworld=\"{}\" -o \"{}\"'.format(
            mitsuba_bash, mts_shadow_xml, samples, cam[0], cam[1], cam[2], ibl_file, ground_path, model_file, world, shadow_out_file)

        final_cmd = '\"{}\" {} -Dw=256 -Dh=256 -Dsamples={} -q -Dori=\"{}\" -Dtarget=\"{}\" -Dup=\"{}\" -Dibl=\"{}\" -Dground={}\" -Dmodel=\"{}\" -Dworld=\"{}\" -o \"{}\"'.format(
            mitsuba_bash, mts_final_xml, samples, cam[0], cam[1], cam[2], ibl_file, ground_path, model_file, world, final_out_file)
    
    # with open('test.txt','w+') as f:
    #     f.write(shadow_cmd)
    #     f.write('\n')
    #     f.write(final_cmd)

    mitsuba_util_bash = '/home/ysheng/Documents/mitsuba/dist/mtsutil'
    shadow_tonemapping_cmd = '{} tonemap {}'.format(mitsuba_util_bash,shadow_out_file)
    if real_ibl:
        tone_scale = 5.0
    else:
        ibl_np = imageio.imread(ibl_file)
        if ibl_np.dtype == np.uint8:
            ibl_np = ibl_np/255.0

        tone_scale = 4 * np.sum(ibl_np)
    final_tonemapping_cmd = '{} tonemap -m {} {}'.format(mitsuba_util_bash, tone_scale, final_out_file)

    if write_cmd:
        with open(cmd_path, 'a+') as f:
            if final:
                f.write('{}\n{}\n{}\n{}\n'.format(shadow_cmd, final_cmd, shadow_tonemapping_cmd, final_tonemapping_cmd))
            else:
                f.write('{}\n{}\n'.format(shadow_cmd, shadow_tonemapping_cmd))
    else:
        # os.system(shadow_cmd)
        return_code = subprocess.check_output(shadow_cmd, shell=True) 
        if options.verbose:
            print(return_code) 

        # os.system(mts_util_tonemapping_cmd)
        return_code = subprocess.check_output(shadow_tonemapping_cmd, shell=True) 
        if options.verbose:
            print(return_code) 

        # os.system(final_cmd)
        return_code = subprocess.check_output(final_cmd, shell=True) 
        if options.verbose:
            print(return_code) 


        # os.system(mts_util_tonemapping_cmd)
        return_code = subprocess.check_output(final_tonemapping_cmd, shell=True) 
        if options.verbose:
            print(return_code) 
        print('mitsuba finshed')

def net_gt(mask_file, ibl_file, out_file):
    # given mask, get bases
    dirname = os.path.basename(os.path.dirname(mask_file))
    pitch_rot = os.path.splitext(os.path.basename(mask_file))[0]
    pitch_rot = pitch_rot[:pitch_rot.find('_mask')]

    ibl_base_folder = join('/home/ysheng/Dataset/new_dataset/base/', dirname)
    shadow_base_file = join(ibl_base_folder, pitch_rot + '_shadow.npy')

    # use ibl to compute the gt shadow
    shadow_bases = np.load(shadow_base_file)
    h, w, iw, ih = shadow_bases.shape
    
    out_dir = os.path.dirname(out_file)
    ibl_fname = os.path.splitext(os.path.basename(ibl_file))[0]
    
    # remember, this ibl should always be 80 x 512
    # ibl = np.load(ibl_file)
    ibl = to_net_ibl(ibl_file)    
    save_ibl = np.copy(ibl)
    cv2.normalize(ibl, save_ibl, 0.0, 1.0, cv2.NORM_MINMAX)
    plt.imsave(join(out_dir, "ibl.png"), save_ibl, cmap='gray')

    ibl = cv2.flip(ibl, 0)

    # ibl = cv2.resize(ibl, (iw, ih), interpolation=cv2.INTER_NEAREST)
    shadow = np.tensordot(shadow_bases, ibl, axes=([2,3], [1,0]))

    # save
    np.save(out_file, shadow)
    if options.verbose:
        print('net gt finish')
    dirname, fname = os.path.dirname(out_file), os.path.splitext(os.path.basename(out_file))[0]
    png_output = os.path.join(dirname, fname + '.png')
    cv2.normalize(shadow, shadow, 0.0, 1.0, cv2.NORM_MINMAX)
    plt.imsave(png_output, shadow, cmap='gray')

    return shadow


def net_render(mask_file, ibl_file, out_file, save_npy=True):
    s = time.time()
    net_ibl = to_net_ibl(ibl_file)
    if net_ibl.shape[0] != 5 and net_ibl.shape[1] != 32:
        print('net render ibl is wrong, please check: ', net_ibl.shape)
        return None

    ibl = cv2.flip(net_ibl, 0)
    mask_np = imageio.imread(mask_file)
    mask_np = mask_np[:,:,0]
    if mask_np.dtype == np.uint8:
        mask_np = mask_np/255.0

    mask, ibl = to_tensor(np.expand_dims(mask_np, axis=2)), to_tensor(np.expand_dims(cv2.resize(ibl, (32,16)), axis=2))
    with torch.no_grad():
        I_s, L_t = torch.unsqueeze(mask.to(device),0), torch.unsqueeze(ibl.to(device),0)

        predicted_img, predicted_src_light = model(I_s, L_t)

    shadow_predict = np.squeeze(predicted_img[0].detach().cpu().numpy().transpose((1,2,0)))
    if save_npy:
        dirname, fname = os.path.dirname(out_file), os.path.splitext(os.path.basename(out_file))[0]
        npy_out = os.path.join(dirname, fname + '.npy')
        np.save(npy_out, shadow_predict)
    
    dirname, fname = os.path.dirname(out_file), os.path.splitext(os.path.basename(out_file))[0]
    png_output = os.path.join(dirname, fname + '.png')
    cv2.normalize(shadow_predict, shadow_predict, 0.0, 1.0, cv2.NORM_MINMAX)
    plt.imsave(png_output, shadow_predict, cmap='gray')
    
    if options.verbose:
        print('net predict {} finished, time: {}s'.format(out_file, time.time() -s))
    
    return shadow_predict

def merge_result(rendered_img, mask_file, shadow_img, out_file):
    pass

def evaluate(mask_file, ibl_file, output, real_ibl=True):
    """ output/mitsuba_final.png
        output/mitsuba_shadow.png
        output/mitsuba_merge.png
        output/prediction_shadow.npy
        output/net_gt_shadow.npy
        output/predcition_merge.npy
    """
    mitsuba_final = join(output, 'mitsuba_final.exr')
    mitsuba_shadow_output, mitsuba_merge = join(output, 'mitsuba_shadow.exr'), join(output, 'mitsuba_merge.png')
    net_shadow_output, net_merge = join(output, 'prediction_shadow.npy'), join(output, 'prediction_merge.png')
    net_gt_output, net_gt_merge = join(output, 'net_gt_shadow.npy'), join(output, 'net_gt_merge.npy')

    # call mitsuba render 
    mitsuba_render(mask_file, ibl_file, mitsuba_final, mitsuba_shadow_output, real_ibl)

    # call net render result
    dirname, fname = os.path.dirname(mask_file), os.path.splitext(os.path.basename(mask_file))[0]
    mask_file = join(dirname, fname + ".npy")
    net_render(mask_file, ibl_file, net_shadow_output)
    net_gt(mask_file, ibl_file, net_gt_output)
    
    # merge result
    # merge_result(mitsuba_final, )


if __name__ == '__main__':
    # # evaluate(options.file, options.mask, options.ibl, options.output)
    # model_file = '/Data_SSD/models/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A.obj'
    # mask_file = '/Data_SSD/new_dataset/cache/mask/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/pitch_15_rot_0_mask.npy'
    # # ibl_file = '/home/ysheng/Dataset/ibls/real/20060430-01_hd.hdr'
    # # ibl_file = '../test_pattern.png'
    

    # ibl_file = '../test_pattern.png'

    # # model_file = '/home/ysheng/Dataset/models/simulated_combine_female_short_outfits_audrey_blair_summertimefull_Base_Pose_Standing_A/simulated_combine_female_short_outfits_audrey_blair_summertimefull_Base_Pose_Standing_A.obj'
    # mask_file = '/home/ysheng/Dataset/new_dataset/cache/mask/simulated_combine_female_long_fullbody_bridget8_wildwind_ssradclosedrobe_CDIG8Female_StandH/pitch_35_rot_0_mask.npy'
    # ibl_file = '/home/ysheng/Dataset/ibls/pattern/num_6_size_0.08_ibl.png'
    # output = '/home/ysheng/Dataset/evaluation/simulated_combine_female_short_outfits_audrey_blair_summertimefull_Base_Pose_Standing_A/num_6_size_0.08_ibl'
    # evaluate(mask_file, ibl_file, output, False)
    
    output = 'dbg/'
    os.makedirs(output, exist_ok=True)
    mask_file = '/Data_SSD/new_dataset/cache/mask/simulated_combine_female_short_outfits_audrey_blair_summertimefull_Base_Pose_Standing_A/pitch_15_rot_0_mask.png'
    shutil.copy(mask_file, 'dbg')
    ibl_file = 'paper/eg/1_ibl.png'
    # ibl_files = ['paper/render/1_ibl.png', 'paper/render/2_ibl.png', 'paper/render/multi_ibl.png']
    ibl_files = ['paper/render/018.png']
    for ibl_file in ibl_files:
        name = os.path.splitext(os.path.basename(ibl_file))[0]    
        out_name = join(output, name + "_net_gt.png")
        out_ibl = join(output, os.path.basename(ibl_file))
        shutil.copy(ibl_file, out_ibl)

        final_out_file = join(output, name + "_final.png")
        shadow_out_file = join(output, name + "_mts_shadow.png")
        mitsuba_render(mask_file, ibl_file, final_out_file, shadow_out_file, final=True, update_cam_param=False, real_ibl=False, write_cmd=False, skip=False)
        net_gt(mask_file, ibl_file, out_name)
