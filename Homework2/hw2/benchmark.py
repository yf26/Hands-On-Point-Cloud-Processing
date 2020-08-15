# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct
from scipy import spatial

import script.octree as octree
import script.kdtree as kdtree
from script.result_set import KNNResultSet, RadiusNNResultSet


def read_velodyne_bin(path):
    """
    :param path:
    :return: homography matrix of the point cloud, N*3
    """
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


# if __name__ == '__main__':
def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    root_dir = '000000.bin'  # 数据集路径
    points_raw = read_velodyne_bin(root_dir)

    # random_indices = np.random.randint(0, points_raw.shape[0], 10000)
    random_indices = list(range(10000))
    db_np = points_raw[random_indices]

    print("octree --------------")
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0

    begin_t = time.time()
    root = octree.octree_construction(db_np, leaf_size, min_extent)
    construction_time_sum = time.time() - begin_t

    iteration_num = 10000
    for i in range(iteration_num):
        query = db_np[i, :]

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        octree.octree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        octree.octree_radius_search(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Octree: build %.3fms, knn %.3fms, radius %.3fms, brute %.3fms" % (construction_time_sum * 1000,
                                                                             knn_time_sum * 1000 / iteration_num,
                                                                             radius_time_sum * 1000 / iteration_num,
                                                                             brute_time_sum * 1000 / iteration_num))

    print("kdtree --------------")
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0

    begin_t = time.time()
    root = kdtree.kdtree_construction(db_np, leaf_size)
    construction_time_sum = time.time() - begin_t

    for i in range(iteration_num):
        query = db_np[i, :]

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        kdtree.kdtree_radius_search(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Kdtree: build %.3fms, knn %.3fms, radius %.3fms, brute %.3fms" % (construction_time_sum * 1000,
                                                                             knn_time_sum * 1000 / iteration_num,
                                                                             radius_time_sum * 1000 / iteration_num,
                                                                             brute_time_sum * 1000 / iteration_num))

    print("scipy.spatial.KDTree --------------")
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0

    begin_t = time.time()
    kdtree_sp = spatial.KDTree(db_np, leafsize=leaf_size)
    construction_time_sum = time.time() - begin_t

    for i in range(iteration_num):
        query = db_np[i, :]

        begin_t = time.time()
        # result_set = KNNResultSet(capacity=k)
        d, i = kdtree_sp.query(query, k)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result = kdtree_sp.query_ball_point(query, radius)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Kdtree: build %.3fms, knn %.3fms, radius %.3fms, brute %.3fms" % (construction_time_sum * 1000,
                                                                             knn_time_sum * 1000 / iteration_num,
                                                                             radius_time_sum * 1000 / iteration_num,
                                                                             brute_time_sum * 1000 / iteration_num))


if __name__ == '__main__':
    main()
