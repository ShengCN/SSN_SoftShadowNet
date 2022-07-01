import os
from os.path import join
from glob import glob
import h5py
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


IBLW, IBLH = 64, 16
def fill_dset(scene_prefix: str, dset):
    folder = os.path.dirname(scene_prefix)
    prefix = os.path.basename(scene_prefix)

    iblis = [8 * i for i in range(IBLW)]
    ibljs = [128 + 8 * i for i in range(IBLH)]

    for i, ibli in enumerate(tqdm(iblis, desc='Build for {}'.format(scene_prefix))):
        for j, iblj in enumerate(ibljs):
            shadow_file = '{}_ibli_{}_iblj_{}_shadow.png'.format(scene_prefix, ibli, iblj)
            dset[:,:,i,j] = plt.imread(shadow_file)[..., 0]


def render_hdf5(opt:dict):
    cache    = opt['cache']
    width    = opt['width']
    height   = opt['height']
    out_hdf5 = opt['out_hdf5']

    if os.path.exists(out_hdf5):
        raise ValueError('{} has existed. Delete it? '.format(out_hdf5))

    scenes = glob(join(cache, '**/*_mask.png'), recursive=True)
    scenes = [scene[:scene.find('_mask.png')] for scene in scenes]

    print(scenes)

    with h5py.File(out_hdf5, 'w') as f:
        for scene in tqdm(scenes, desc='Build hdf5'):
            scene_name = 'base/{}_{}'.format(os.path.basename(os.path.dirname(scene)), os.path.basename(scene))
            dset = f.create_dataset(scene_name, (width, height, IBLW, IBLH), chunks=(width, height, 1, 1), dtype='f', compression="gzip")
            fill_dset(scene, dset)

            input_name = 'x/{}_{}'.format(os.path.basename(os.path.dirname(scene)), os.path.basename(scene))
            dset = f.create_dataset(input_name, (width, height, 2), chunks=(width, height, 2), dtype='f', compression="gzip")

            mask_name = '{}_mask.png'.format(scene)
            ao_name = '{}_touch.png'.format(scene)

            dset[:,:,0] = plt.imread(mask_name)[:,:,0]
            dset[:,:,1] = plt.imread(ao_name)[:,:,0]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=str, help='csv')
    parser.add_argument('--width', type=int, help='image width')
    parser.add_argument('--height', type=int, help='image height')
    parser.add_argument('--out_hdf5', type=str, help='output hdf5 path')
    params = parser.parse_args()

    print(params)

    render_hdf5(vars(params))
