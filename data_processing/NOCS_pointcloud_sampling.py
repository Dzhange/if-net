import implicit_waterproofing as iw
from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
from glob import glob
import os,re
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import random
import traceback
import cv2

ROOT = None
sample_num = None

def find_frame_num(path):
    return re.findall(r'%s(\d+)' % "frame_",path)[0]

def NOCS_point_sampling(path):
    nocs_path = glob(os.path.join(path, '*nox00_01pred.png'))
    # mask_path = glob(os.path.join(path, '*predmask.png'))
    if len(nocs_path) > 1:
        print("[ERROR] more than one input")
        print(nocs_path)
    nocs_path = nocs_path[0]
    # mask_path = mask_path[0]
    nocs = cv2.imread(nocs_path)
    nocs = cv2.cvtColor(nocs,cv2.COLOR_BGR2RGB)
    
    valid_idx = np.where(np.all(nocs != [255, 255, 255], axis=-1)) # Only white BG
    num_valid = valid_idx[0].shape[0]
    # for current use we choose uniform sample
    sampled_idx = (valid_idx[0][np.random.choice(num_valid,sample_num,replace=False)] ,
                    valid_idx[1][np.random.choice(num_valid,sample_num,replace=False)])
    sample_points = nocs[sampled_idx[0], sampled_idx[1]] / 255
    return sample_points

def nocs_pointcloud_sampling(path,translation,scale):
    try:
        out_file = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(args.res, args.num_points)

        if os.path.exists(out_file):
            if args.write_over:
                print("Overwriting ",out_file)
            else:
                print("file existed: ",out_file,", skip.")
                return

        point_cloud = NOCS_point_sampling(path)
        point_cloud += translation
        point_cloud *= scale

        if args.noise:
            point_cloud = point_cloud + 0.01 * np.random.randn(point_cloud.shape[0], 3)
        occupancies = np.zeros(len(grid_points), dtype=np.int8)
        
        _, idx = kdtree.query(point_cloud)
        occupancies[idx] = 1
        compressed_occupancies = np.packbits(occupancies)

        np.savez(out_file, point_cloud=point_cloud, 
          compressed_occupancies = compressed_occupancies, 
          bb_min = bb_min, bb_max = bb_max, 
          res = args.res)
        print('Finished {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

def scale_nocs(path):

    frame = find_frame_num(path)
    mesh_name = "frame_" + frame + "_NOCS_mesh.obj"
    target_name = "isosurf_scaled.off"
    
    if os.path.exists(os.path.join(path,target_name)):
        if args.write_over:
            print('overwrite ', os.path.join(path,target_name))
        else:
            print('File exists. Done.')
            return

    translation = 0
    scale = 1

    try:
        mesh = trimesh.load(os.path.join(path,mesh_name), process=False)
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) /2
        
        translation = -centers
        scale = 1/total_size

        mesh.apply_translation(-centers)
        mesh.apply_scale(1/total_size)
        mesh.export(path + '/isosurf_scaled.off')
    except Exception as e:
        print('Error {} with {}'.format(e,path))
    print('Finished {}'.format(path))

    return translation, scale


# First we 
def scale_sample(path):
    translation, scale = scale_nocs(path)
    nocs_pointcloud_sampling(path,translation,scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run point cloud sampling'
    )

    parser.add_argument('--res', type=int, default=128)
    parser.add_argument('--num-points', type=int,default=3000)
    parser.add_argument('--noise',type=bool,default=False)
    parser.add_argument('--write-over',type=bool,default=False)
    parser.add_argument('--input-dir',type=str,default='/ZG/nocs_data_ifnet/val')
    args = parser.parse_args()

    bb_min = -0.5
    bb_max = 0.5

    grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, args.res)
    kdtree = KDTree(grid_points)

    sample_num = args.num_points

    # test_path = "/ZG/frame_00000000_view_00_test"
    
    
    # scale_sample(test_path)
    # scale(test_path)

    # nocs_pointcloud_sampling(test_path)
    
    
    p = Pool(mp.cpu_count())
    paths = glob(os.path.join(args.input_dir, '*'))

    # enabeling to run te script multiple times in parallel: shuffling the data
    # random.shuffle(paths)
    # print(paths,len(paths))
    p.map(scale_sample, paths)