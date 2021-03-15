import pickle
import os
from os.path import join
import glob
from tqdm import tqdm

model_dict = {}
with open('model_prf.bin', 'rb') as f:
    model_dict = pickle.load(f)

model_root = '/home/ysheng/Dataset/general_models'
out_folder = '/home/ysheng/Dataset/general_ds_render/'

model_path_dict = {}
models = glob.glob(join(model_root, '*'))
for m in models:
    mname = os.path.splitext(os.path.basename(m))[0]
    model_path_dict[mname] = m

out_str = []
# generate scripts
for k in tqdm(model_dict.keys()):
    triple_list = model_dict[k]
    for t in triple_list:
        p,r,fov = t
        out_path = join(out_folder, 'cache/shadow_output', k)
        cmd = 'build/hard_shadow_fov --model={} --output={} --gpu={} --resume={} --cam_pitch={} --model_rot={} --fov={} --render_touch \n'.format(model_path_dict[k], out_path, 0, True, p, r, fov)
        out_str.append(cmd)

with open("general_touch.sh", 'w+') as f:
    for i in out_str[:len(out_str)//3]:
        f.write(i) 

with open("general_touch1.sh", 'w+') as f:
    for i in out_str[len(out_str)//3:len(out_str)//3 * 2]:
        f.write(i) 

with open("general_touch2.sh", 'w+') as f:
    for i in out_str[len(out_str)//3 * 2:]:
        f.write(i) 