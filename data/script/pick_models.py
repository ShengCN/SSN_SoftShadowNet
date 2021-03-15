import os
from os.path import join
import glob
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import random
import csv

category_map = {0: 'human', 1: 'general'}
def pick_models(model_path, num, csv_file, category=0, replace=False):
    support_ext = ['*.obj', '*.off']
    files = []
    for ext in support_ext:
        files.extend(glob.glob(join(model_path, ext)))

    num = min(num, len(files))
    print('Picking {} files'.format(num))

    picked_files = random.sample(files, num)

    # write to csv
    open_hint = 'a'
    if replace:
        open_hint = 'w'

    with open(csv_file, open_hint) as f:
        writer = csv.writer(f, delimiter=',')
        for f in tqdm(picked_files):
            writer.writerow([f, '{}'.format(category)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pick models from ')
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--category", default=0, type=int,
                        help="0 is human, 1 is general model")
    parser.add_argument("--num", type=int, help="# of files selected")
    parser.add_argument("--csv", type=str, help='Output csv file')
    parser.add_argument("--replace", action='store_true', help='Replace current csv file')
    args = parser.parse_args()

    print('parameters: {}'.format(args))
    pick_models(args.model_path, args.num, args.csv, category=args.category, replace=args.replace)
