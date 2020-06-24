import os,sys
import glob
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import shutil
import argparse
import re

output_dir = None
script_path = None

pred_res_path = None
output_dir = None

def find_frame_num(path):
    return re.findall(r'%s(\d+)' % "frame_",path)[0]
 
    # Then, cluster all the corresponding VIEWS in to the folder

def create_folders_proc(frame):
    
    view_num = 10
    for view_idx in range(view_num):
        frame_and_view = "frame_" + frame + "_view_" + str(view_idx).zfill(2)
        sub_dir = os.path.join(output_dir, frame_and_view)
        if os.path.exists(sub_dir):
            continue
        print("working on {}".format(frame_and_view))
        cur_files = glob.glob(os.path.join(pred_res_path, frame_and_view) + '*')
        os.mkdir(sub_dir)
        for f in cur_files:
            f_name = os.path.basename(f)
            new_f = os.path.join(sub_dir,f_name)
            if not os.path.exists(new_f):
                shutil.copy(f,new_f)
                # print("copied {} to {}".format(f,new_f))
        
            # /ZG/meshedNOCS/hand_rig_dataset_v3/train/0000/frame_00000000_NOCS_mesh.obj
        nocs_name = "frame_" + frame + "_NOCS_mesh.obj"
        nocs_path = os.path.join(pred_res_path,nocs_name)
        new_nocs_path = os.path.join(sub_dir, nocs_name)
        shutil.copy(nocs_path,new_nocs_path)

def create_folders():
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print(pred_res_path)
    # first, get all FRAMES in the target folder
    all_frames = [find_frame_num(p) for p in glob.glob(os.path.join(pred_res_path,'*color00.png'))]
    max_frame = max(all_frames)
    # print(all_frames)
    all_frames = list(dict.fromkeys(all_frames))
    all_frames.sort()
    
    p = Pool(mp.cpu_count() >> 3)
    p.map(create_folders_proc, all_frames)


# def scale(path):

#     frame = find_frame_num(path)
#     mesh_name = "frame_" + frame + "_NOCS_mesh.obj"
#     target_name = "isosurf_scaled.off"
    
#     print(path)
#     if os.path.exists(os.path.join(path,target_name)):
#         if args.write_over:
#             print('overwrite ', os.path.join(path,target_name))
#         else:
#             print('File exists. Done.')
#             return

#     try:
#         mesh = trimesh.load(os.path.join(path,mesh_name), process=False)
#         total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
#         centers = (mesh.bounds[1] + mesh.bounds[0]) /2

#         mesh.apply_translation(-centers)
#         mesh.apply_scale(1/total_size)
#         mesh.export(path + '/isosurf_scaled.off')
#     except:
#         print('Error with {}'.format(path))
#     print('Finished {}'.format(path))

# Mesh_dataset_path should be the path to hand
# def get_paired_mesh(mesh_dataset_path,pred_res_path):


#     all_objs = glob.glob( os.path.join(mesh_dataset_path,'*/*NOCS_mesh.obj'))

    
#     all_frames = [ find_frame_num(p) for p in glob.glob(os.path.join(pred_res_path,'*color00.png'))]
#     Max_frame = max(all_frames)
#     print(Max_frame)
#     for obj in all_objs:
#         mesh_name = os.path.basename(obj)
#         mesh_frame = find_frame_num(mesh_name)
#         new_path = os.path.join(pred_res_path,mesh_name)
        
#         if not os.path.exists(new_path) and Max_frame >= mesh_frame:
#             print(new_path)
#             shutil.copy(obj,new_path)


def genPredResults(nocs_dir):
    
    os.chdir(nocs_dir)

    for i in range(5):
        base_command = "python nrnocs.py \
        --mode test \
        --output-dir ../../nrnocs_output \
        --out-targets nox00 \
        --data-limit 30 \
        --val-data-limit 5 \
        --downscale 2 \
        --input-dir ../../meshedNOCS --gpu 3 \
        --batch-size 2 \
        --expt-name NRNOCS_NOX00_b2_d20_wtBG \
        --if-net True\
        --test-input /ZG/meshedNOCS/hand_rig_dataset_v3/train/{}".format( str(i).zfill(4) )

        os.system(base_command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Close hole, densify, and convert to .off format for Mano dataset'
    )

    parser.add_argument('--write-over',default=False,type=bool,help="Overwrite previous results if set to True")
    parser.add_argument('--pred-res',default='../nrnocs_output/NRNOCS_NOX00_b2_d20_wtBG/TestResults/', help='Provide the directory while the prediction result of nrnocs is stored')
    parser.add_argument('--hand-dataset',default='../meshedNOCS/hand_rig_dataset_v3/train/',help='Path to the original dataset, while GT mesh is stored')
    parser.add_argument('--output-dir',default='/ZG/nocs_data_ifnet' ,help='Provide the output directory where the processed Mano Model(in .off format would be stored)')
    parser.add_argument('--script-path', default="./data_processing/closehole_densify_iter3.mlx")
    parser.add_argument('--nocs-dir',default='/ZG/CatRecon/nrnocs', help='the working directory of nrnocs')
    parser.add_argument('--densify', default=False, type=bool)
    args = parser.parse_args()

    script_path = args.script_path
    pred_res_path = args.pred_res
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(script_path):
        print("[ERROR] No Usable Script found")
        sys.exit()

    # all_input = glob.glob( os.path.join(args.input_dir,'*'))
    
    # genPredResults(args.nocs_dir)

    # create_folders()

    all_input = glob.glob( os.path.join(args.output_dir,'*'))

    p = Pool(mp.cpu_count())

    p.map(scale, all_input)

    # # densify is optional, since the GT mesh for NOCS is much denser than Mano
    # if not args.densify:
    #     script_path = ''

    # p.map(close_hole_densify_to_off, glob.glob(args.output_dir + '/*'))

    # p.map(scale, glob.glob(args.output_dir + '/*'))