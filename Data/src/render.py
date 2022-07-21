import os
from os.path import join
from glob import glob
from tqdm import tqdm
import pandas as pd
import argparse
import multiprocessing
import shutil
import random


def worker(input):
    gpu, model_path, out_path, w, h, cam_pitch, model_rot = input
    os.makedirs(out_path, exist_ok=True)

    cam_pitch_str = '{}'.format(cam_pitch)
    model_rot_str = '{}'.format(model_rot)

    # for p in cam_pitch:
    #     cam_pitch_str = cam_pitch_str + ',' + p

    # for r in model_rot:
    #     model_rot_str = model_rot_str + ',' + r

    # cam_pitch_str = cam_pitch_str[1:]
    # model_rot_str = model_rot_str[1:]

    cmd = 'build/shadow_base --verbose=0 --model={} --output={} --cam_pitch={} --model_rot={} --gpu={} --render_mask --render_shadow --render_touch --width={} --height={} --ibl_w=512 --ibl_h=256 --base_avg'.format(model_path, out_path, cam_pitch_str, model_rot_str, gpu, w, h)
    print(cmd)
    os.system(cmd)


def render_raw_imgs(params):
    start_id      = params.start_id
    end_id        = params.end_id
    cpus          = params.cpus
    gpus          = params.gpus
    width         = params.width
    height        = params.height
    model_path    = params.model_root
    samples       = params.samples
    cam_pitch_min = params.cam_pitch_min
    cam_pitch_max = params.cam_pitch_max
    model_rot_min = params.model_rot_min
    model_rot_max = params.model_rot_max
    output_folder = params.out_folder

    random.seed(19920208)

    all_models = glob(join(model_path, '*.obj')) + glob(join(model_path, '*.off'))
    if end_id == -1:
        all_models = all_models[start_id:]
    else:
        all_models = all_models[start_id:end_id]

    processer_num = cpus
    inputs        = []

    for i, model in enumerate(all_models):
        for si in range(samples):
            model_output = join(output_folder, os.path.splitext(os.path.basename(model))[0])

            rand_camera_pitch = random.uniform(cam_pitch_min, cam_pitch_max)
            rand_model_pitch  = random.uniform(model_rot_min, model_rot_max)

            cur_ind = i * samples + si
            inputs.append([gpus[cur_ind%len(gpus)], model, model_output, width, height, rand_camera_pitch, rand_model_pitch])


    import pdb; pdb.set_trace()

    task_num = len(inputs)
    with multiprocessing.Pool(processer_num) as pool:
        for i, _ in tqdm(enumerate(pool.imap_unordered(worker, inputs), 1), total=task_num, desc='Render'):
            # print("Finished: {} \r".format(float(i) / task_num), flush=True, end='')
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpus', type=int, help='how many cpu?')
    parser.add_argument('--gpus', nargs="+", type=int, help='which gpu?')
    parser.add_argument('--model_root', type=str, help='Model path')
    parser.add_argument('--out_folder', type=str, help='output folder')
    parser.add_argument('--start_id', type=int, help='start index')
    parser.add_argument('--end_id', type=int, help='end index')

    parser.add_argument('--width', type=int, help='base width')
    parser.add_argument('--height', type=int, help='base height')

    parser.add_argument('--samples', type=int, help='samples per scene')
    parser.add_argument('--cam_pitch_min', type=float, help='minimum camera pitch')
    parser.add_argument('--cam_pitch_max', type=float, help='maximum camera pitch')
    parser.add_argument('--model_rot_min', type=float,  help='minimum model rotation')
    parser.add_argument('--model_rot_max', type=float, help='maximum model rotation')


    params = parser.parse_args()


    print('Input params: ')
    print(params)

    render_raw_imgs(params)
