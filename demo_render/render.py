import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import random

def get_files(folder):
    return [join(folder,f) for f in os.listdir(folder) if os.path.isfile(join(folder, f))]

def render_img(path, light, out_folder):
    fname = os.path.splitext(os.path.basename(path))[0]
    light_name = os.path.splitext(os.path.basename(light))[0]

    out_path = join(out_folder,"{}".format(fname))
    # import pdb; pdb.set_trace()

    cmd = 'mitsuba {} -o {} -Dsample_num={} -Denv_path={} -Dw=512 -Dh=512 -q'.format(path, out_path, 1024, light)
    print(cmd)
    os.system(cmd)

def render_mitsuba(folder, light_folder):
    folders = glob.glob(join(folder, "*"))
    light_files = glob.glob(join(light_folder, "*.png"))
    print(len(folders), len(light_files))
    # import pdb; pdb.set_trace()

    for f in tqdm(folders):
        files = glob.glob(join(f, "*.xml"))
        for file in tqdm(files):
            cur_light = random.sample(light_files,k=1)[0]
            render_img(file, cur_light, f)

            cmd = 'mtsutil tonemap {}'.format(join(f, "*.exr"))
            os.system(cmd)

if __name__ == '__main__':
    root = "/home/ysheng/Documents/paper_project/adobe/user_study_data/"
    random.seed(19920208)
    render_mitsuba(join(root, 'mts_cache/general'),"/home/ysheng/Documents/paper_project/adobe/user_study_data/lights")
    render_mitsuba(join(root, 'mts_cache/human'),"/home/ysheng/Documents/paper_project/adobe/user_study_data/lights")
