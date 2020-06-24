import trimesh
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import glob
import os
import argparse
from voxels import VoxelGrid

def create_voxel_off(path):

    pc_path = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(args.res, args.num_points)
    off_pc_path = path + '/voxelized_point_cloud_{}res_{}points.off'.format(args.res, args.num_points)
    off_voxel_path = path + '/voxelized_point_cloud_{}res_{}voxel.off'.format(args.res, args.num_points)

    pc = np.load(pc_path)['point_cloud']
    trimesh.Trimesh(vertices = pc , faces = []).export(off_pc_path)


    ## create voxel .off file, copied from 'create_voxel_off.py'
    occ = np.unpackbits(np.load(pc_path)['compressed_occupancies'])
    voxels = np.reshape(occ, (args.res,)*3)
    min = -0.5
    max = 0.5
    loc = ((min+max)/2, )*3
    scale = max - min
    VoxelGrid(voxels, loc, scale).to_mesh().export(off_voxel_path)

    print('Finished: {}'.format(path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create off visualization from point cloud.'
    )

    parser.add_argument('-res', type=int)
    parser.add_argument('-num_points', type=int)

    args = parser.parse_args()

    ROOT = 'shapenet/data'

    create_voxel_off("/ZG/frame_00000000_view_00_test")
    # p = Pool(mp.cpu_count())
    # p.map(create_voxel_off, glob.glob(ROOT + '/*/*/'))