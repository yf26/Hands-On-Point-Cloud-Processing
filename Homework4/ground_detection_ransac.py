# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类
import time

import numpy as np
import math
import os
import struct
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from ground_detection_SVD import extract_initial_seeds, pcd_preprocessing


# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    ground_indices = ransac_on_segments(data)
    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    # print('segmented data points num:', segmengted_cloud.shape[0])
    print('segmented data points num:', ground_indices.shape[0])
    return ground_indices


def ransac_on_segments(data, segment_x=0, max_iteration=40, threshold=0.15):
    """
    ransac on 2 segments in x direction
    :param data:
    :param segment_x: given the partition position segment_x
    :param max_iteration:
    :param threshold:
    :return:
    """
    total_inidices = np.array(range(data.shape[0]))

    forward_filter = data[:, 0] >= segment_x
    forward_indices = total_inidices[forward_filter]
    forward_data = data[forward_filter]
    backward_indices = total_inidices[np.logical_not(forward_filter)]
    backward_data = data[np.logical_not(forward_filter)]

    inliers_idx1, _ = my_ransac(forward_data, forward_indices, max_iteration, threshold)
    inliers_idx2, _ = my_ransac(backward_data, backward_indices, max_iteration, threshold)
    return np.r_[inliers_idx1, inliers_idx2]


def ransac_on_segments_v2(data, segments_num=5, max_iteration=40, threshold=0.15):
    """
    ransac on several uniformly distributed segments in x direction
    :param data:
    :param segments_num: given the number of segments segments_num
    :param max_iteration:
    :param threshold:
    :return:
    """
    total_inidices = np.array(range(data.shape[0]))
    x_min = np.min(data[:, 0])
    x_max = np.max(data[:, 0])
    seg_interval = (x_max - x_min) / segments_num
    seg_bound = x_min + np.array(range(segments_num+1)) * seg_interval
    print(seg_bound)

    stacked_ground_idx = np.empty(0, dtype=int)
    for i in range(segments_num):
        range_filter = np.logical_and(data[:, 0] < seg_bound[i+1], data[:, 0] > seg_bound[i])
        filtered_data = data[range_filter]
        filtered_indices = total_inidices[range_filter]
        ground_idx, _ = my_ransac(filtered_data, filtered_indices, max_iteration, threshold)
        if ground_idx is None:
            continue
        stacked_ground_idx = np.r_[stacked_ground_idx, ground_idx]
    return stacked_ground_idx


def my_ransac(data, indices, max_iteration, threshold):
    assert (data.shape[0] == indices.shape[0])

    # select points near the ground
    # z_high = -1.73 + 0.3
    # z_low = -1.73 - 0.3
    # z_filter = np.logical_and(data[:, 2] < z_high, data[:, 2] > z_low)

    # # down sample selected pcd
    # z_filtered_pcd = o3d.geometry.PointCloud()
    # z_filtered_pcd.points = o3d.utility.Vector3dVector(data[z_filter, :])
    # # o3d.visualization.draw_geometries([z_filtered_pcd])
    # z_filtered_pcd = z_filtered_pcd.uniform_down_sample(20)
    # filtered_data = np.asarray(z_filtered_pcd.points)
    # print("filtered data size = {}".format(filtered_data.shape))
    # if filtered_data.shape[0] < 40:
    #     print("No ground in this segement!")
    #     return None, None
    # # o3d.visualization.draw_geometries([z_filtered_pcd])

    # visualize_pcd(data)
    filtered_data = extract_initial_seeds(data, 40000, 1)
    # visualize_pcd(filtered_data)

    best_set_size = 0
    best_model_params = []
    iteration = 0
    while iteration < max_iteration:
        rng = np.random.default_rng()
        selected_points_indices = rng.choice(range(filtered_data.shape[0]), 3, replace=False)
        selected_points = filtered_data[selected_points_indices, :]
        params = estimate_plane_params(selected_points)

        # calculate inliers
        dists = np.fabs(np.c_[filtered_data, np.ones((filtered_data.shape[0], 1))].dot(params))
        inliers_num = np.sum([dists < threshold])
        if inliers_num > best_set_size:
            best_set_size = inliers_num
            best_model_params = params
            # e = inliers_num / total_points_num
            # max_iteration = math.floor(math.log2(1 - p) / math.log2(1 - math.pow((1 - e), 3)))
            # print("max_iteration = {}".format(max_iteration))
        iteration += 1

    print("Total iterations = {}".format(iteration))
    print("Best model params = {}".format(best_model_params))

    # apply found params on origin pcd
    dists_on_all = np.fabs(np.c_[data, np.ones((data.shape[0], 1))].dot(best_model_params))
    inliers_idx = indices[dists_on_all < threshold]

    return inliers_idx, best_model_params


def estimate_plane_params(selected_points: np.array):
    vector1 = selected_points[1, :] - selected_points[0, :]
    vector2 = selected_points[2, :] - selected_points[0, :]

    # calculate plane params
    a = (vector1[1] * vector2[2]) - (vector1[2] * vector2[1])
    b = (vector1[2] * vector2[0]) - (vector1[0] * vector2[2])
    c = (vector1[0] * vector2[1]) - (vector1[1] * vector2[0])
    d = -(a * selected_points[0, 0] + b * selected_points[0, 1] + c * selected_points[0, 2])
    n = math.sqrt(a ** 2 + b ** 2 + c ** 2)
    params = np.array([a / n, b / n, c / n, d / n])
    return params


# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始

    # 屏蔽结束

    return clusters_index


# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection='3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()


def visualize_pcd(points, rgb=None):
    pcd_seg = o3d.geometry.PointCloud()
    pcd_seg.points = o3d.utility.Vector3dVector(points)
    if rgb is not None:
        pcd_seg.paint_uniform_color(rgb)
    o3d.visualization.draw_geometries([pcd_seg])


def main():
    root_dir = 'data/'  # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[1:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        segmented_points = ground_segmentation(data=origin_points)
        cluster_index = clustering(segmented_points)

        plot_clusters(segmented_points, cluster_index)


if __name__ == '__main__':
    # main()
    path = "test/000111.bin"
    points = read_velodyne_bin(path)
    points = pcd_preprocessing(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 1, 0])

    start = time.time()
    ground_indices = ground_segmentation(points)
    print("Ground estimation takes {} ms".format((time.time() - start) * 1000))
    ground = points[ground_indices]

    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(ground)
    ground_pcd.paint_uniform_color([0.8, 0.8, 0.8])

    o3d.visualization.draw_geometries([pcd, ground_pcd])
