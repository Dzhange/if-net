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

sample_num = None
GT = False
both_sides = False # use 2 views(00,01) of ground truth as input

def find_frame_num(path):
    return re.findall(r'%s(\d+)' % "frame_",path)[0]

def NOCS_point_sampling(path):
    
    nocs_path = []
    frame_view = os.path.basename(path)
    if GT:
        nocs_path.append(os.path.join(path, frame_view+'_nox00.png'))
        if both_sides: 
            nocs_path.append(os.path.join(path, frame_view+'_nox01.png'))
    else:
        nocs_path = glob(os.path.join(path, '*nox00_01pred.png'))
        if both_sides: 
            nocs_path = glob(os.path.join(path, '*nox00_01pred.png'))

    all_points = None
    for p in nocs_path:
        nocs = cv2.imread(p)
        nocs = cv2.cvtColor(nocs,cv2.COLOR_BGR2RGB)
        
        valid_idx = np.where(np.all(nocs != [255, 255, 255], axis=-1)) # Only white BG
        num_valid = valid_idx[0].shape[0]
        
        randomIdx = np.random.choice(num_valid,sample_num // len(nocs_path),replace=False)
        # for current use we choose uniform sample
        sampled_idx = (valid_idx[0][randomIdx], valid_idx[1][randomIdx])
        sample_points = nocs[sampled_idx[0], sampled_idx[1]] / 255
        
        if all_points is None:
            all_points = sample_points
        else:
            all_points = np.concatenate((all_points, sample_points), axis=0)
        print(all_points.shape)
    return all_points

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

def scale_sample(path):
    translation, scale = scale_nocs(path)
    nocs_pointcloud_sampling(path,translation,scale)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run point cloud sampling'
    )

    parser.add_argument('--res', type=int, default=128)
    parser.add_argument('--num-points', type=int,default=30000)
    parser.add_argument('--noise',type=bool,default=False)
    parser.add_argument('--write-over',type=bool,default=False)
    parser.add_argument('--GT',type=bool,default=False)
    parser.add_argument('--mode',type=str,default='train')
    parser.add_argument('--input-dir',type=str,default='/ZG/nocs_gt_ifnet/')
    parser.add_argument('--both_sides',type=bool,default=True)
    
    args = parser.parse_args()

    bb_min = -0.5
    bb_max = 0.5

    grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, args.res)
    kdtree = KDTree(grid_points)

    sample_num = args.num_points
    GT = args.GT
    both_sides = args.both_sides

    if 0:
        test_path = "/ZG/frame_00000000_view_00"
        scale_sample(test_path)
        # scale(test_path)
        # nocs_pointcloud_sampling(test_path)
    else:
        p = Pool(mp.cpu_count()>>3)
        paths = glob(os.path.join(args.input_dir,args.mode, '*'))
        # enabeling to run te script multiple times in parallel: shuffling the data
        # random.shuffle(paths)
        print(paths,len(paths))
        p.map(scale_sample, paths)