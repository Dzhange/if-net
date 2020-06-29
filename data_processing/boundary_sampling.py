import trimesh
import numpy as np
import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback

ROOT = 'shapenet/data'


def boundary_sampling(path):
    try:
        out_file = path +'/boundary_{}_samples.npz'.format(args.sigma)

        if os.path.exists(out_file):
            if args.write_over:
                print('overwrite ', out_file)
            else:
                print('File exists. Done.')
                return

        off_path = path + '/isosurf_scaled.off'
        
        mesh = trimesh.load(off_path)
        points = mesh.sample(sample_num)

        boundary_points = points + args.sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords

        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

        np.savez(out_file, points=boundary_points, occupancies = occupancies, grid_coords= grid_coords)
        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-sigma', type=float, default=0.1)
    parser.add_argument('-write_over',type=bool,default=True)
    parser.add_argument('-input-dir',type=str,default='/ZG/nocs_gt_ifnet/')
    parser.add_argument('-mode',type=str,default='train')
    args = parser.parse_args()


    sample_num = 100000

    # test_path = "/ZG/frame_00000000_view_00_test"
    # boundary_sampling(test_path)
    p = Pool(mp.cpu_count() >> 2)
    paths = glob.glob(os.path.join( args.input_dir,args.mode,'*'))
    
    p.map(boundary_sampling, paths)
