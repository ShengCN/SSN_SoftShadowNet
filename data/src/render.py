import os
from os.path import join
from glob import glob
from tqdm import tqdm
import pandas as pd
import argparse
import multiprocessing
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', nargs="+", type=int, help='which gpu?')
parser.add_argument('--ds_root', type=str, help='Dataset path')
parser.add_argument('--out_folder', type=str, help='output folder')
parser.add_argument('--start_id', type=int, help='start index')
parser.add_argument('--end_id', type=int, help='end index')
params = parser.parse_args()

def worker(input):
    gpu, model_path, out_path = input
    os.makedirs(out_path, exist_ok=True)
    cmd = 'build/shadow_base --model={} --output={} --cam_pitch=30 --model_rot=-45 --render_mask --render_height --render_normal --render_shadow --gpu={} --width=512 --height=512 --ibl_w=512 --ibl_h=256'.format(model_path, out_path, gpu)
    os.system(cmd)

all_models = glob(join(params.ds_root, '*'))[params.start_id:params.end_id]
processer_num = len(params.gpus)
inputs = []
for i in range(processer_num):
    delta = int(len(all_models)/processer_num)
    cur_pro_models = all_models[i * delta:(i+1)*delta]
    if i == processer_num - 1:
        cur_pro_models = all_models[i * delta:]

    for model in cur_pro_models:
        model_output = join(params.out_folder, os.path.splitext(os.path.basename(model))[0])
        inputs.append([params.gpus[i], model, model_output])

task_num = len(inputs)
with multiprocessing.Pool(processer_num) as pool:
    for i, _ in enumerate(pool.imap_unordered(worker, inputs), 1):
        print("Finished: {} \r".format(float(i) / task_num), flush=True, end='')

