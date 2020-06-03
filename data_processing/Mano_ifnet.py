import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import shutil


# INPUT_PATH = '/home/lky/ZG/GuibasLab/data/Mano/handsOnly_REGISTRATIONS_r_l'
INPUT_PATH = '/home/lky/ZG/GuibasLab/data/ifnet_data/Mano/handsOnly_testDataset_REGISTRATIONS/'

def create_folders(path):

    if not os.path.isfile(path) or path[-3:] != "ply":  
        return

    file_name = os.path.basename(path)
    file_prefix = file_name.split('.')[0]
    dir_name = os.path.join(INPUT_PATH, file_prefix)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        target_name = os.path.join(dir_name, file_name)
        shutil.move(path, target_name)
    

all_objs = glob.glob( INPUT_PATH + '/*')
p = Pool(mp.cpu_count())
p.map(create_folders, all_objs)

def close_hole_densify_to_off(path):

    if os.path.exists(path + '/isosurf.off'):
        return

    input_file  = os.path.join(path, os.path.basename(path) +'.ply')
    print("input file is ",input_file)
    output_file =os.path.join(path, 'isosurf.off')

    script_path = "/home/lky/ZG/GuibasLab/data/Mano/mano_closehole_densify.mlx"

    cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {} -o {} -s {}'.format(input_file,output_file,script_path)
    # if you run this script on a server: comment out above line and uncomment the next line
    # cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {} -o {}'.format(input_file,output_file)
    os.system(cmd)

p = Pool(mp.cpu_count())
p.map(close_hole_densify_to_off, glob.glob( INPUT_PATH + '/*'))

def scale(path):

    if os.path.exists(path + '/isosurf_scaled.off'):
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

p = Pool(mp.cpu_count())
p.map(scale, glob.glob( INPUT_PATH + '/*'))
