import os
from os.path import join
from glob import glob
from tqdm import tqdm
import pandas as pd
import argparse
import multiprocessing
import shutil


def worker(input):
    gpu, model_path, out_path, w, h, cam_pitch, model_rot = input
    os.makedirs(out_path, exist_ok=True)

    cam_pitch_str = ''
    model_rot_str = ''

    for p in cam_pitch:
        cam_pitch_str = cam_pitch_str + ',' + p

    for r in model_rot:
        model_rot_str = model_rot_str + ',' + r

    cam_pitch_str = cam_pitch_str[1:]
    model_rot_str = model_rot_str[1:]

    cmd = 'build/shadow_base --model={} --output={} --cam_pitch={} --model_rot={} --gpu={} --render_mask --render_shadow --render_touch --width={} --height={} --ibl_w=512 --ibl_h=256 --base_avg'.format(model_path, out_path, cam_pitch_str, model_rot_str, gpu, w, h)
    print(cmd)
    os.system(cmd)


def render_raw_imgs(params):
    all_models    = glob(join(params.ds_root, '*.obj')) + glob(join(params.ds_root, '*.off'))
    if params.end_id == -1:
        all_models = all_models[params.start_id:]
    else:
        all_models = all_models[params.start_id:params.end_id]

    # processer_num = len(params.gpus)
    processer_num = params.cpus
    inputs        = []

    # for i in range(processer_num):
    #     delta          = int(len(all_models)/processer_num)
    #     cur_pro_models = all_models[i * delta:(i+1)*delta]

    #     if i == processer_num - 1:
    #         cur_pro_models = all_models[i * delta:]

    #     for model in cur_pro_models:
    #         model_output = join(params.out_folder, os.path.splitext(os.path.basename(model))[0])
    #         inputs.append([params.gpus[i%len(params.gpus)], model, model_output, params.width, params.height, params.cam_pitch, params.model_rot])

    for i, model in enumerate(all_models):
        model_output = join(params.out_folder, os.path.splitext(os.path.basename(model))[0])
        inputs.append([params.gpus[i%len(params.gpus)],
                       model,
                       model_output,
                       params.width,
                       params.height,
                       params.cam_pitch,
                       params.model_rot])


    import pdb; pdb.set_trace()

    task_num = len(inputs)
    with multiprocessing.Pool(processer_num) as pool:
        for i, _ in enumerate(pool.imap_unordered(worker, inputs), 1):
            print("Finished: {} \r".format(float(i) / task_num), flush=True, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpus', type=int, help='how many cpu?')
    parser.add_argument('--gpus', nargs="+", type=int, help='which gpu?')
    parser.add_argument('--ds_root', type=str, help='Dataset path')
    parser.add_argument('--out_folder', type=str, help='output folder')
    parser.add_argument('--start_id', type=int, help='start index')
    parser.add_argument('--end_id', type=int, help='end index')

    parser.add_argument('--width', type=int, help='base width')
    parser.add_argument('--height', type=int, help='base height')
    parser.add_argument('--cam_pitch', nargs="+", help='camera pitch')
    parser.add_argument('--model_rot', nargs="+", help='model rotation ')

    params = parser.parse_args()

    render_raw_imgs(params)
