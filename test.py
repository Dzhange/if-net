import numpy as np

cam2world = np.array([[1,0,0,0],
    [ 0,1,0,0],
    [0,0,1,1],
    [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=np.float32)


# cam2world = np.array([[ 0.85408425,  0.31617427, -0.375678  ,  0.56351697 * 2],
#     [ 0.        , -0.72227067, -0.60786998,  0.91180497 * 2],
#     [-0.52013469,  0.51917219, -0.61688   ,  0.92532003 * 2],
#     [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=np.float32)
world2cam = np.linalg.inv(cam2world).astype('float32')
X_homo = np.array([1,0,0,1])
print(cam2world)
print(world2cam @ X_homo)
