import os,sys
import glob
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import shutil
import argparse


output_dir = None
script_path = None

def create_folders(mesh_path):

    if not os.path.isfile(mesh_path) or mesh_path[-3:] != "ply":  
        return

    file_name = os.path.basename(mesh_path)
    file_prefix = file_name.split('.')[0]
    dir_name = os.path.join(output_dir, file_prefix)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        target_name = os.path.join(dir_name, file_name)
        shutil.move(mesh_path, target_name)
    
def close_hole_densify_to_off(dir_path):
    
    output_file =os.path.join(dir_path, 'isosurf.off')
    
    if os.path.exists(output_file):
        if args.write_over:
            print('overwrite ', output_file)
        else:
            print('File exists. Done.')
            return

    input_file  = os.path.join(dir_path, os.path.basename(dir_path) +'.ply')
    print("input file is ",input_file)    

    cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {} -o {} -s {}'.format(input_file,output_file,script_path)
    # if you run this script on a server: comment out above line and uncomment the next line
    # cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {} -o {}'.format(input_file,output_file)
    os.system(cmd)

def scale(path):

    if os.path.exists(path + '/isosurf_scaled.off'):
        if args.write_over:
            print('overwrite ', path + '/isosurf_scaled.off')
        else:
            print('File exists. Done.')
            return

    try:
        mesh = trimesh.load(path + '/isosurf.off', process=False)
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) /2

        mesh.apply_translation(-centers)
        mesh.apply_scale(1/total_size)
        mesh.export(path + '/isosurf_scaled.off')
    except:
        print('Error with {}'.format(path))
    print('Finished {}'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Close hole, densify, and convert to .off format for Mano dataset'
    )

    parser.add_argument('--write-over',default=False,type=bool,help="Overwrite previous results if set to True")
    parser.add_argument('--input-dir',default='./shapenet/data/train_raw/', help='Provide the input directory where datasets are stored.(Mano/)')
    parser.add_argument('--output-dir',default='./shapenet/data/train' ,help='Provide the output directory where the processed Mano Model(in .off format would be stored)')
    parser.add_argument('--script-path', default="./data_processing/closehole_densify_iter3.mlx")
    args = parser.parse_args()

    output_dir = args.output_dir
    script_path = args.script_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(script_path):
        print("[ERROR] No Usable Script found")
        sys.exit()

    all_input = glob.glob( os.path.join(args.input_dir,'*'))
    
    p = Pool(mp.cpu_count())

    p.map(create_folders, all_input)

    p.map(close_hole_densify_to_off, glob.glob(args.output_dir + '/*'))

    p.map(scale, glob.glob(args.output_dir + '/*'))