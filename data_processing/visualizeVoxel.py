import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
import numpy as np

if __name__ == "__main__":

    parser.add_argument('--input', type=str)
    args = parser.parse_args()

    voxel_res = 32
    voxel = np.load(args.input)
    voxel = np.reshape(voxel_res, voxel_res, voxel_res)
    voxel = np.unpackbits(voxel)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.voxel(voxel)

    plt.show()