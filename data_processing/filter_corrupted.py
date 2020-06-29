import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import shutil
import numpy
import numpy as np

# python data_processing/filter_corrupted.py -file 'voxelization_32.npy'

def filter(path):

    if not os.path.exists('{}/{}'.format(path,file)):
        print('Remove: {}'.format(path, file))
        if delete:
            shutil.rmtree('{}'.format(path))


parser = argparse.ArgumentParser(
    description='Filter shapenet objects if preprocessing failed.'
)
parser.add_argument('-file', type=str)
parser.add_argument('-delete', action='store_true')
parser.add_argument('-input-dir',type=str,default='/ZG/nocs_gt_ifnet/')
parser.add_argument('--mode',type=str,default='train')
parser.set_defaults(delete=False)

args = parser.parse_args()
file = args.file
delete = args.delete

# ROOT = 'shapenet/data/'

p = Pool(mp.cpu_count())
p.map(filter, glob.glob(os.path.join(args.input_dir, args.mode,'*')))

def update_split():

    split = np.load('shapenet/split.npz')
    split_dict = {}
    for set in ['train','test','val']:
        filterd_set = split[set].copy()
        for path in split[set]:
            if not os.path.exists('shapenet/data/' + path):
                print('Filtered: ' + path)
                filterd_set = np.delete(filterd_set, np.where(filterd_set == path))
        split_dict[set] = filterd_set

    np.savez('shapenet/split.npz', **split_dict)


if delete:
    update_split()