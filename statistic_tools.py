import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


def split_data(data):
    return


def get_historgram(img_path, min_depth=0, max_depth=655.5):
    depth = np.asarray(Image.open(img_path))
    depth = decode_apollo(np.resize(depth, (513, 513, 3)))
    plt.hist(depth, bins=655, range=(min_depth, max_depth), density=True, histtype='step')
    plt.ylabel('Probability')
    # plt.show()


def decode_apollo(depth_im):
    R = depth_im[:, :, 0] / 255
    G = depth_im[:, :, 1] / 255
    real_depth = (R + G / 255.0) * 655.36
    return real_depth


def depth_to_pointcloud(depth, calib_matrix, save_path='points.txt'):
    points = []
    k_inv = np.linalg.inv(calib_matrix)
    xyz = np.zeros((depth.shape[0], depth.shape[1], 3))
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            p = depth[i, j] * k_inv.dot(np.array([j, i, 1]))
            points.append(p)
            xyz[i, j] = p
    points = np.array(points)
    np.savetxt(save_path, points)
    print("point cloud created...\n")
    return xyz


if __name__ == '__main__':
    # root = r"D:\Shai_Schneider\Apollo_dataset\val\Depth"
    # files = recursive_glob(root)
    # random_images = np.random.choice(len(files), size=len(files) // 10, replace=False)
    #
    # for file in random_images:
    #     get_historgram(files[file])
    # # p = r"D:\Shai_Schneider\Apollo_dataset\val\Depth\09-00\CLEAR_SKY\DEGRADATION\With_Pedestrian\With_TrafficBarrier\Urban_Straight_Road\Traffic_093\0000000.png"
    # # get_historgram(p, min_depth=0, max_depth=656)
    # # p = r"D:\Shai_Schneider\Apollo_dataset\val\Depth\09-00\CLEAR_SKY\DEGRADATION\With_Pedestrian\With_TrafficBarrier\Downtown\Traffic_093\0000005.png"
    # # get_historgram(p, min_depth=0, max_depth=656)
    # # p=r"D:\Shai_Schneider\Apollo_dataset\val\Depth\09-00\CLEAR_SKY\DEGRADATION\With_Pedestrian\With_TrafficBarrier\Downtown\Traffic_095\0000019.png"
    # # get_historgram(p, min_depth=0, max_depth=656)
    # # p=r"D:\Shai_Schneider\Apollo_dataset\val\Depth\09-00\CLEAR_SKY\DEGRADATION\With_Pedestrian\With_TrafficBarrier\Downtown\Traffic_095\0000059.png"
    # # get_historgram(p, min_depth=0, max_depth=656)
    # plt.show()
    img_path_predicted = r"D:\Shai_Schneider\pytorch-deeplab-xception\examples\example_2stage.png"
    img_path_gt = r"D:\Shai_Schneider\Apollo_dataset\test\Depth\13-00\CLEAR_SKY\NO_DEGRADATION\With_Pedestrian\With_TrafficBarrier\Downtown\Traffic_117\0000000.png"
    depth = np.asarray(Image.open(img_path_predicted))
    depth = decode_apollo(depth)
    K = [[2015.0, 0, 960.0],
         [0, 2015.0, 540.0],
         [0, 0, 1]]
    xyz_pred = depth_to_pointcloud(depth, K, 'two_stage_points.txt')

    depth = np.asarray(Image.open(img_path_gt))
    depth = decode_apollo(depth)
    K = [[2015.0, 0, 960.0],
         [0, 2015.0, 540.0],
         [0, 0, 1]]
    xyz_gt = depth_to_pointcloud(depth, K, 'gt_points.txt')

    l1 = np.linalg.norm(xyz_gt-xyz_pred)
