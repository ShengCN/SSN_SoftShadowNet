import os
from os.path import join
import glob
from tqdm import tqdm
import random

def generate():
    def one_folder(folder, out_folder, base=False):
        models = glob.glob(join(folder,'*'))
        print('{} has {} models'.format(folder, len(models)))
        str = ''

        for i, model in enumerate(tqdm(models)):
            cam_pitch = [int(random.random() * 30.0) for i in range(2)]
            model_rot = [int((random.random()* 2.0-1.0) * 90.0) for i in range(2)]
            cam_pitch_str, model_rot_str = '{},{}'.format(cam_pitch[0], cam_pitch[1]), '{},{}'.format(model_rot[0], model_rot[1])

            if not base:
                cmd = 'python render_dataset.py --gpu=0 --num={} --start_id={} --end_id={} --resume --cam_pitch={} --model_rot={} --model_folder=\"{}\" --out_folder=\"{}\" \n'.format(len(models), i, i+1, cam_pitch_str, model_rot_str, folder, out_folder)
            else:
                cmd = 'python render_dataset.py --gpu=0 --num={} --start_id={} --end_id={} --resume --base --cam_pitch={} --model_rot={} --model_folder=\"{}\" --out_folder=\"{}\" \n'.format(len(models), i, i+1, cam_pitch_str, model_rot_str, folder, out_folder)
            str += cmd
        
        return str


    root = '/home/ysheng/Dataset/benchmark_ds/models'
    out_root = '/home/ysheng/Dataset/benchmark_ds/net_shadow_gt'
    human_folder = join(root, 'human')
    general_folder = join(root, 'general')
    
    random.seed(19920208)
    cmds = one_folder(human_folder, join(out_root, 'human'))
    cmds += one_folder(general_folder, join(out_root, 'general'))

    random.seed(19920208)
    cmds += one_folder(human_folder, join(out_root, 'human'), base=True)
    cmds += one_folder(general_folder, join(out_root, 'general'), base=True)
    
    with open('render_benchmark.sh', 'w+') as f:
        f.write(cmds)
    
    print('finished')
    

if __name__ == '__main__':
    generate()