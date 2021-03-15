import os
from os.path import join
from shutil import copyfile
from tqdm import tqdm

def get_files(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(join(folder, f))]

def get_folders(folder):
    return [f for f in os.listdir(folder) if not os.path.isfile(join(folder, f))]


if __name__ == '__main__':
    folders = get_folders('/home/ysheng/Dataset/new_dataset/base')
    model_folder = '/home/ysheng/Dataset/human_models'
    models = get_files(model_folder)

    out_folder = '../human_models'
    os.makedirs(out_folder,exist_ok=True)
    for f in tqdm(folders):
        model_path = join(model_folder, f + '.obj')
        dst_path = join(out_folder, f)
        copyfile(model_path, dst_path)