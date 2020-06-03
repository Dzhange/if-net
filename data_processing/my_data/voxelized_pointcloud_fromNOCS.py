# This file transfer NOCS data into point cloud and occupancies.

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/home/lky/ZG/GuibasLab/data/hand_rig_dataset_v3')
    parser.add_argument('-o', '--output', type=str, default='shapenet/data/')
    parser.add_argument('-res', type=int)
    parser.add_argument('-num_points', type=int)
    
    args = parser.parse_args()

    