# 实现voxel滤波，并加载数据集中的文件进行验证

import time
import random
import open3d as o3d
import os
import pandas as pd
import math
import numpy as np
from pyntcloud import PyntCloud


# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, random_pick=False):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    point_cloud = np.asarray(point_cloud)
    x_max = max(point_cloud[:, 0])
    y_max = max(point_cloud[:, 1])
    x_min = min(point_cloud[:, 0])
    y_min = min(point_cloud[:, 1])
    z_min = min(point_cloud[:, 2])
    print("x_min = {}, x_max = {}".format(x_min, x_max))
    Dx = math.ceil((x_max - x_min) / leaf_size)
    Dy = math.ceil((y_max - y_min) / leaf_size)

    h_list = []
    for i in range(point_cloud.shape[0]):
        hx = math.floor((point_cloud[i, 0] - x_min) / leaf_size)
        hy = math.floor((point_cloud[i, 1] - y_min) / leaf_size)
        hz = math.floor((point_cloud[i, 2] - z_min) / leaf_size)
        h_list.append([i, hx + hy * Dx + hz * Dx * Dy])
    h_list.sort(key=lambda x: x[1])

    # centroid
    if not random_pick:
        temp = h_list[0][1]
        idx_collection = []
        for i in range(point_cloud.shape[0]):
            if h_list[i][1] == temp:
                idx_collection.append(h_list[i][0])
            else:
                temp = h_list[i][1]
                filtered_points.append(np.sum(point_cloud[idx_collection, :], axis=0) / len(idx_collection))
                idx_collection.clear()
                idx_collection.append(h_list[i][0])
    else:
        temp = h_list[0][1]
        idx_collection = []
        for i in range(point_cloud.shape[0]):
            if h_list[i][1] == temp:
                idx_collection.append(h_list[i][0])
            else:
                temp = h_list[i][1]
                filtered_points.append(point_cloud[random.choice(idx_collection), :])
                idx_collection.clear()
                idx_collection.append(h_list[i][0])

    # 屏蔽结束
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    file_name = '/home/yu/3D Point Cloud Processing/DataSet/MyDataSet/car/test/car_0221.ply'
    point_cloud_pynt = PyntCloud.from_file(file_name)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云

    # start = time.time()
    # point_cloud_o3d = point_cloud_o3d.voxel_down_sample(voxel_size=100)
    # print("Down sample takes {}s.".format(time.time() - start))

    # 调用voxel滤波函数，实现滤波
    print("Random voxel downsampling...")
    start = time.time()
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 0.25, random_pick=True)
    if filtered_cloud.shape[0] == 0:
        print("Please select proper leaf size!")
        return None
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    print("Down sample takes {}s.".format(time.time() - start))
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

    # 调用voxel滤波函数，实现滤波
    print("Centroid voxel downsampling...")
    start = time.time()
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 0.25, random_pick=False)
    if filtered_cloud.shape[0] == 0:
        print("Please select proper leaf size!")
        return None
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    print("Down sample takes {}s.".format(time.time() - start))
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
