import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import shutil
from tk3dv.ptTools import ptUtils
from tk3dv.ptTools.loaders.GenericImageDataset import GenericImageDataset
from tk3dv.nocstools import datastructures as ds 
import cv2
import numpy as np

# INPUT_PATH = '/home/lky/ZG/GuibasLab/data/Mano/handsOnly_REGISTRATIONS_r_l'
# INPUT_PATH = '/home/lky/ZG/GuibasLab/data/ifnet_data/Mano/handsOnly_testDataset_REGISTRATIONS/'
INPUT_PATH = '/home/lky/ZG/GuibasLab/data/valres/gt'


def create_folders(path):

    if not os.path.isfile(path) or path[-3:] != "png":  
        return

    file_name = os.path.basename(path)
    file_prefix = file_name.split('.')[0]
    dir_name = os.path.join(INPUT_PATH, file_prefix)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        target_name = os.path.join(dir_name, 'nocs.png')
        shutil.move(path, target_name)
    

all_objs = glob.glob( INPUT_PATH + '/*')
p = Pool(mp.cpu_count())
p.map(create_folders, all_objs)


def NOCS_to_scaled_pc(path):
    
    if os.path.exists(path + '/nocs_pc_scaled.npy'):
        return


    nocs = cv2.imread(path + '/nocs.png')
    nocs = cv2.cvtColor(nocs, cv2.COLOR_BGR2RGB)

    nocs = ds.NOCSMap(nocs)
    Points = nocs.Points

    total_size = (Points.max(axis=0) - Points.min(axis=0)).max()
    centers = (Points.max(axis=0) - Points.min(axis=0)) /2
    Points -= centers
    Points /= total_size

    np.save(path + '/nocs_pc_scaled.npy',Points)

p = Pool(mp.cpu_count())
p.map(NOCS_to_scaled_pc, glob.glob(INPUT_PATH + '/*'))


