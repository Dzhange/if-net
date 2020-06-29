
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
test_input_dir = None
mode = None
merge = False

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
    
    
        cur_fixs = ['_color00.png','_color00_nox00_01pred.png','_color00_nox00_03predmask.png']
        os.mkdir(sub_dir)
        for fix in cur_fixs:
            # f_name = os.path.basename(f)
            f_name = frame_and_view + fix
            old_f = os.path.join(pred_res_path,f_name)
            new_f = os.path.join(sub_dir,f_name)
            if os.path.exists(old_f) and not os.path.exists(new_f):
                shutil.copy(old_f,new_f)
                # print("copied {} to {}".format(f,new_f))
        
            # /ZG/meshedNOCS/hand_rig_dataset_v3/train/0000/frame_00000000_NOCS_mesh.obj
        nocs_name = "frame_" + frame + "_NOCS_mesh.obj"
        nocs_path = os.path.join(pred_res_path,nocs_name)
        new_nocs_path = os.path.join(sub_dir, nocs_name)
        shutil.copy(nocs_path,new_nocs_path)

        for suffix in ["_nox00.png","_nox01.png","_pnnocs00.png","_pnnocs01.png"]:
            gt_name = frame_and_view + suffix
            gt_path= os.path.join(test_input_dir,str(int(frame) // 100).zfill(4),gt_name)
            new_gt_path = os.path.join(sub_dir,gt_name)
            shutil.copy(gt_path,new_gt_path)

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
    
    p = Pool(mp.cpu_count() >> 2)
    p.map(create_folders_proc, all_frames)

# Generates the predicted results into nrnocs Output Dir
# seperated into train and val
def genPredResults(nocs_dir,test_input_dir,nrnocs_res_dir,expt_name):
    
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
         --gpu 2\
        --batch-size 2 \
        --expt-name {}\
        --if-net True\
        --test-input {}".format(nrnocs_res_dir,expt_name, cur_input)
        
        # --input-dir ../../meshedNOCS\
        os.system(base_command)


    # move to train / val subdir
    if not os.path.exists(pred_res_path):
        os.mkdir(pred_res_path)
    cur_res_path = os.path.join(args.nrnocs_res, expt_name,"TestResults","frame_00*")
    os.system('mv {} {}'.format(cur_res_path, pred_res_path))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate NOCS MAP from NRNOCS, and prpare the data for If-net'
    )

    parser.add_argument('--write-over',default=False,type=bool,help="Overwrite previous results if set to True")
    parser.add_argument('--nrnocs-res',default='/ZG/nrnocs_output/', help='Function as Input Dir. Provide the directory while the prediction result of nrnocs is stored')
    parser.add_argument('--expt-name', nargs='+',default="NRNOCS_NOX00_b2_d20_wBG", help='the expt name of target model,if more than one, we merge all results for the same target')
    parser.add_argument('--output-dir',default='/ZG/nocs_gt_ifnet' ,help='Provide the output directory where the processed Model')
    parser.add_argument('--nocs-dir',default='/ZG/CatRecon/nrnocs', help='the working directory of nrnocs')
    parser.add_argument('--orig-dataset',default='/ZG/meshedNOCS/hand_rig_dataset_v3/', help='the root of dataset as input for nocs')
    parser.add_argument('--mode', default='train', type=str)

    # expt_name = "NR-NOCS_10percent_BN_batch2" #need 2 GPU
    args = parser.parse_args()
    
    mode = args.mode
    write_over = args.write_over
    output_dir = os.path.join(args.output_dir, mode)
    test_input_dir = os.path.join(args.orig_dataset, args.mode)
    

    # Merge the outputs of multiple model(nox00, nox01), get a full view model
    
    if len(args.expt_name) > 1:
        merge = True
    expt_name = args.expt_name

    pred_res_path = os.path.join(args.nrnocs_res, expt_name,"TestResults/{}".format(mode))
    # pred_res_path = os.path.join(args.nrnocs_res, mode)

    
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    genPredResults(args.nocs_dir,test_input_dir,args.nrnocs_res,expt_name)
    create_folders()
