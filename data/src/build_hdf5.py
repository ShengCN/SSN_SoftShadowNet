import os
from os.path import join
from glob import glob
import h5py
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
import logging
import numpy as np
import pickle

def read_img(inputs):
    fname, i, j = inputs
    try:
        if os.path.exists(fname):
            img = plt.imread(fname)[..., 0]
        else:
            img = None

        return img, i, j
    except BaseException as err:
        logging.error(err)
        return None, i, j


IBLW, IBLH = 64, 16
def fill_dset(scene_prefix:str, dset) -> bool:
    folder = os.path.dirname(scene_prefix)
    prefix = os.path.basename(scene_prefix)

    iblis = [8 * i for i in range(IBLW)]
    ibljs = [128 + 8 * i for i in range(IBLH)]

    # inputs = [['{}_ibli_{}_iblj_{}_shadow.png'.format(scene_prefix, ibli, iblj), ibli, iblj] for ibli in iblis for iblj in ibljs]
    # processer_num = 32
    # with multiprocessing.Pool(processer_num) as pool:
    #     for i, data in tqdm(enumerate(pool.imap_unordered(read_img, inputs), 1), total=len(inputs), desc='Build for {}'.format(scene_prefix)):
    #         try:
    #             img, i, j = data

    #             if img is None:
    #                 logging.info('{} has problem: {},{}'.format(scene_prefix, i, j))
    #                 break

    #             if img is not None:
    #                 dset[:, :, i/8, (j-128)/8] = img

    #         except BaseException as err:
    #             logging.error(err)
    #             break

    for i, ibli in enumerate(iblis):
        for j, iblj in enumerate(ibljs):
            try:
                shadow_file = '{}_ibli_{}_iblj_{}_shadow.png'.format(scene_prefix, ibli, iblj)
                dset[:,:,i,j] = plt.imread(shadow_file)[..., 0]
            except BaseException as err:
                logging.error(err)
                return False
    return True


def resize_ibl(ibl_64_16, ibl_32_8):
    for i in range(IBLW//2):
        for j in range(IBLH//2):
            ibl_32_8[:,:,i,j] = np.sum(ibl_64_16[:,:, 2 * i: 2 * (i+1), 2 * j:2*(j+1)], axis=(2,3))/(2 * 2)



def build_scene_hdf5_worker(inputs):
    scene, width, height, tmp_hdf5_folder = inputs

    scene_basename = '{}_{}'.format(os.path.basename(os.path.dirname(scene)), os.path.basename(scene))
    out_hdf5 = join(tmp_hdf5_folder, scene_basename + '.hdf5')

    try:
        with h5py.File(out_hdf5, 'w') as f:
            # 512 x 512 x 64 x 16 IBL
            scene_name = 'base_64_16/{}'.format(scene_basename)
            base_dset = f.create_dataset(scene_name, (width, height, IBLW, IBLH), chunks=(width, height, 1, 1), dtype='f', compression="gzip")
            succ = fill_dset(scene, base_dset)
            if not succ:
                logging.info('{} failed. We skip'.format(scene_name))
                del f[scene_name]
                return False

            input_name = 'x/{}'.format(scene_basename)
            x_dset = f.create_dataset(input_name, (width, height, 2), chunks=(width, height, 2), dtype='f', compression="gzip")

            mask_name = '{}_mask.png'.format(scene)
            ao_name = '{}_touch.png'.format(scene)

            x_dset[:,:,0] = plt.imread(mask_name)[:,:,0]
            x_dset[:,:,1] = plt.imread(ao_name)[:,:,0]

            # 512 x 512 x 32 x 8 IBL
            scene_name = 'base_32_8/{}'.format(scene_basename)
            base_dset_32_8 = f.create_dataset(scene_name, (width, height, IBLW//2, IBLH//2), chunks=(width, height, 1, 1), dtype='f', compression="gzip")
            resize_ibl(base_dset, base_dset_32_8)

    except BaseException as err:
        logging.error('{}, {}'.format(err, out_hdf5))

def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

def merge_all_hdf5(out_hdf5:str, tmp_hdf5_folder:str):
    os.makedirs(tmp_hdf5_folder, exist_ok=True)

    h5files = glob(join(tmp_hdf5_folder, '*.hdf5'))

    import pdb; pdb.set_trace()
    with h5py.File(out_hdf5, 'w') as dstf:
        for h5file in tqdm(h5files, desc='Merging hdf5'):
            with h5py.File(h5file, 'r') as srcf:
                paths = get_dataset_keys(srcf)
                for p in paths:
                    srcf.copy(srcf[p], dstf['.'], p)


def render_each_scene_hdf5(opt:dict):
    cache    = opt['cache']
    width    = opt['width']
    height   = opt['height']
    out_hdf5 = opt['out_hdf5']

    if os.path.exists(out_hdf5):
        raise ValueError('{} has existed. Delete it? '.format(out_hdf5))

    tmp_hdf5_folder = join(os.path.dirname(out_hdf5), 'hdf5')
    os.makedirs(tmp_hdf5_folder, exist_ok=True)


    cache_file = 'tmp/scene_cache.bin'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            scenes = pickle.load(f)
    else:
        scenes = glob(join(cache, '**/*_mask.png'), recursive=True)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(scenes, f)

    inputs = [[scene[:scene.find('_mask.png')], width, height, tmp_hdf5_folder] for scene in scenes]


    processer_num = 64
    with multiprocessing.Pool(processer_num) as pool:
        for i, data in tqdm(enumerate(pool.imap_unordered(build_scene_hdf5_worker, inputs), 1), total=len(inputs), desc='Build for each scene'):
            pass

    # merge all hdf5s
    merge_all_hdf5(out_hdf5, tmp_hdf5_folder)


if __name__ == '__main__':
    log_file = 'logs/{}.log'.format(os.path.splitext(__file__)[0])
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s - %(filename)s %(funcName)s %(asctime)s ",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ])

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=str, help='csv')
    parser.add_argument('--width', type=int, help='image width')
    parser.add_argument('--height', type=int, help='image height')
    parser.add_argument('--out_hdf5', type=str, help='output hdf5 path')
    params = parser.parse_args()

    print(params)

    render_each_scene_hdf5(vars(params))
