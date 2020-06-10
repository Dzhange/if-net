import implicit_waterproofing as iw
from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
from glob import glob
import os,sys,shutil
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import random
import traceback
FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '../../pyRender/src'))
sys.path.append(os.path.join(FileDirPath, '../../pyRender/lib')) # add path to pyRender lib
import render
import objloader
import skimage.io as sio
import time

ROOT = 'shapenet/data/'
# This file takes uni-scaled isosurface mesh as input, and output a sampled point,
# directly from mesh. 
# Here we use the pyRender to perform depth culling，get partial point cloud.
def voxelized_pointcloud_sampling(path):
    try:
        out_file = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(args.res, args.num_points)

        if os.path.exists(out_file):
            if args.write_over:
                print('overwrite ', out_file)
            else:
                print('File exists. Done.')
                return
        off_path = path + '/isosurf_scaled.off'

        # mesh = trimesh.load(off_path)
        # point_cloud = mesh.sample(args.num_points)
        V, F = objloader.LoadOff(off_path)
        # set up camera information
        info = {'Height':480, 'Width':640, 'fx':575, 'fy':575, 'cx':319.5, 'cy':239.5}
        render.setup(info)

        # set up mesh buffers in cuda
        context = render.SetMesh(V, F)
        
        cam2world = np.array([[ 0.85408425,  0.31617427, -0.375678  ,  0.56351697 * 2],
            [ 0.        , -0.72227067, -0.60786998,  0.91180497 * 2],
            [-0.52013469,  0.51917219, -0.61688   ,  0.92532003 * 2+3],
            [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=np.float32)

        world2cam = np.linalg.inv(cam2world).astype('float32')
        # the actual rendering process
        render.render(context, world2cam)

        # get information of mesh rendering
        # vindices represents 3 vertices related to pixels
        vindices, vweights, findices = render.getVMap(context, info)
        visible_indices = np.unique(vindices)
        # visible_points = V[visible_indices]
        # visible_faces =  F[np.unique(findices)]
        num_visible = visible_indices.shape[0]
    
        # # faces for partial boun
        # for i in range(num_visible):
        #     origin_index = visible_indices[i]
        #     visible_faces = np.where(visible_faces==origin_index,i,visible_faces)

        # max_index = num_visible - 1
        # faces_valid = np.ones((visible_faces.shape[0]),dtype=bool)
        # for i in range(visible_faces.shape[0]):
        #     face = visible_faces[i]
        #     if face[0] > max_index or face[1] > max_index or face[2] > max_index:
        #         faces_valid[i] = False
        # visible_faces = visible_faces[faces_valid]
        
        
        # for visualization
        # vis_face = findices.astype('float32') / np.max(findices)
        # sio.imsave(path + '/face.png',vis_face)
        # sio.imsave(path + '/vertex.png',vweights)
        # return num_visible

        if num_visible < args.num_points:
            print("visible_num is ",num_visible)
            raise Exception
        sample_idx = np.random.choice(num_visible,args.num_points,replace=False)
        valid_indices = visible_indices[sample_idx]
        point_cloud = V[valid_indices]
        # num_valid = point_cloud.shape[0]
        
        # for i in range(num_valid):
        #     origin_index = valid_indices[i]
        #     visible_faces = np.where(visible_faces==origin_index,i,visible_faces)

        # max_index = num_valid - 1
        # faces_valid = np.ones((visible_faces.shape[0]),dtype=bool)
        # for i in range(visible_faces.shape[0]):
        #     face = visible_faces[i]
        #     if face[0] > max_index or face[1] > max_index or face[2] > max_index:
        #         faces_valid[i] = False
        # visible_faces = visible_faces[faces_valid]

        occupancies = np.zeros(len(grid_points), dtype=np.int8)

        _, idx = kdtree.query(point_cloud)
        occupancies[idx] = 1

        compressed_occupancies = np.packbits(occupancies)

        np.savez(out_file, point_cloud=point_cloud, 
          compressed_occupancies = compressed_occupancies, 
          bb_min = bb_min, bb_max = bb_max, 
          res = args.res)

        print('Finished {}'.format(path))
        
        # prepare data for boundary sampling
        
        # visible_
        # visible_mesh = trimesh.base.Trimesh(vertices=point_cloud,faces=visible_faces)
        # visible_mesh.export(path + '/vis_mesh.obj')
        # boundary_sampling(path,visible_mesh)
    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run point cloud sampling'
    )

    parser.add_argument('-res', type=int)
    parser.add_argument('-num_points', type=int)
    parser.add_argument('-write_over',type=bool)
    # parser.add_argument('-sigma', type=float)

    args = parser.parse_args()
    MIN_NUM = 99999
    bb_min = -0.5
    bb_max = 0.5
    min_num = 999999
    grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, args.res)
    kdtree = KDTree(grid_points)

    p = Pool(mp.cpu_count())
    paths = glob(ROOT + '/*/*/')
    # enabeling to run te script multiple times in parallel: shuffling the data
    random.shuffle(paths)
    p.map(voxelized_pointcloud_sampling, paths)
    # voxelized_pointcloud_sampling('/home/lky/ZG/GuibasLab/if-net/shapenet/data/train/01_01r')
