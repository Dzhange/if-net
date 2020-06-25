import os,sys
import glob
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import shutil
import argparse
import re

pred_res_path = None
output_dir = None
expt_name = None
write_over = False
mode = None

def find_frame_num(path):
    return re.findall(r'%s(\d+)' % "frame_",path)[0]
 
    # Then, cluster all the corresponding VIEWS in to the folder

def create_folders_proc(frame):
    print(frame)
    view_num = 10
    for view_idx in range(view_num):
        frame_and_view = "frame_" + frame + "_view_" + str(view_idx).zfill(2)
        sub_dir = os.path.join(output_dir, frame_and_view)
        if os.path.exists(sub_dir):
            print("{} already exists".format(sub_dir))
            if write_over:
                print("Removing {}".format(sub_dir))
                shutil.rmtree(sub_dir)
            else:
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


def genPredResults(nocs_dir,test_input_dir,nrnocs_res_dir):
    
    os.chdir(nocs_dir)
    
    SUBSETS_IDX = ''
    if mode == 'val':
        SUBSETS_IDX = range(10,12) # 0010 ... 0011
    elif  mode == 'train':
        SUBSETS_IDX = range(0,6) # 0000 ... 0005

    for i in SUBSETS_IDX:

        cur_input = os.path.join(test_input_dir,str(i).zfill(4))
        
        base_command = "python nrnocs.py \
        --mode test \
        --output-dir {} \
        --out-targets nox00 \
        --data-limit 30 \
        --val-data-limit 5 \
        --downscale 2 \
        --input-dir ../../meshedNOCS --gpu 1\
        --batch-size 2 \
        --expt-name {}\
        --if-net True\
        --test-input {}".format(nrnocs_output_dir,expt_name, cur_input)

        os.system(base_command)


    # move to train / val subdir
    os.mkdir(pred_res_path)
    cur_res_path = os.path.join(args.nrnocs_res, expt_name,"TestResults","frame_00*")
    os.system('mv {} {}'.format(cur_res_path, pred_res_path))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate NOCS MAP from NRNOCS, and prpare the data for If-net'
    )

    parser.add_argument('--write-over',default=False,type=bool,help="Overwrite previous results if set to True")
    parser.add_argument('--nrnocs-res',default='../nrnocs_output/', help='Function as Input Dir. Provide the directory while the prediction result of nrnocs is stored')
    parser.add_argument('--expt-name',default="NRNOCS_NOX00_b2_d20_wBG", help='the expt name of target model')
    parser.add_argument('--output-dir',default='/ZG/nocs_data_ifnet' ,help='Provide the output directory where the processed Model')
    parser.add_argument('--nocs-dir',default='/ZG/CatRecon/nrnocs', help='the working directory of nrnocs')
    parser.add_argument('--orig-dataset',default='/ZG/meshedNOCS/hand_rig_dataset_v3/', help='the root of dataset as input for nocs')
    parser.add_argument('--mode', default='train', type=str)

    # expt_name = "NR-NOCS_10percent_BN_batch2" #need 2 GPU
    args = parser.parse_args()

    mode = args.mode
    expt_name =  args.expt_name
    write_over = args.write_over
    pred_res_path = os.path.join(args.nrnocs_res, expt_name,"TestResults/{}".format(mode))
    output_dir = os.path.join(args.output_dir, mode)
    test_input_dir = os.path.join(args.orig_dataset, args.mode)
    

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    
    genPredResults(args.nocs_dir,test_input_dir,args.nrnocs_res)
    create_folders()
