import struct
import time

import numpy as np
import math
import open3d as o3d
from result_set import KNNResultSet, RadiusNNResultSet


class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value  # position of partition plane, none means leaf node
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False


def update_axis(axis, dim):
    if axis == dim - 1:
        return 0
    else:
        return axis + 1


def sort_keys_by_values(point_indices, values):
    assert len(point_indices.shape) == 1
    assert values.shape == point_indices.shape
    idx_sorted = np.argsort(values)
    keys_sorted = point_indices[idx_sorted]
    values_sorted = values[idx_sorted]
    return keys_sorted, values_sorted


def kdtree_recursively_build_fast(root, db: np.ndarray, point_indices: np.ndarray, axis, leaf_size):
    if root is None:  # establish a node X
        root = Node(axis, None, None, None, point_indices)

    assert len(point_indices.shape) == 1
    if point_indices.shape[0] > leaf_size:
        # point_indices_sorted, values_sorted = sort_keys_by_values(point_indices, db[point_indices, axis])
        #
        # middle_left_pos = math.ceil(point_indices_sorted.shape[0] / 2) - 1
        # middle_right_pos = middle_left_pos + 1
        # middle_left_idx = point_indices_sorted[middle_left_pos]
        # middle_right_idx = point_indices_sorted[middle_right_pos]
        #
        # # TODO: find median point value without sorting or using mean value as partition value
        # # set root.value for the node X
        # middle_left_point_val = db[middle_left_idx, axis]
        # middle_right_point_val = db[middle_right_idx, axis]
        # root.value = (middle_left_point_val + middle_right_point_val) / 2

        mean = np.mean(db[point_indices, axis])
        root.value = mean
        print(mean)

        middle_left_indices = []
        middle_right_indices = []

        for i in point_indices:
            if db[i, axis] <= mean:
                middle_left_indices.append(i)
            else:
                middle_right_indices.append(i)

        # build left subtree for node X
        root.left = kdtree_recursively_build(
            root.left, db,
            point_indices[middle_left_indices],
            update_axis(axis, db.shape[1]),
            leaf_size
        )

        # build left subtree for node X
        root.right = kdtree_recursively_build(
            root.right, db,
            point_indices[middle_right_indices],
            update_axis(axis, db.shape[1]),
            leaf_size
        )
    return root


# def kdtree_recursively_build(root, db: np.ndarray, point_indices: np.ndarray, axis, leaf_size):
#     if root is None:  # establish a node X
#         root = Node(axis, None, None, None, point_indices)
#
#     assert len(point_indices.shape) == 1
#     if point_indices.shape[0] > leaf_size:
#         point_indices_sorted, values_sorted = sort_keys_by_values(point_indices, db[point_indices, axis])
#
#         middle_left_pos = math.ceil(point_indices_sorted.shape[0] / 2) - 1
#         middle_right_pos = middle_left_pos + 1
#         middle_left_idx = point_indices_sorted[middle_left_pos]
#         middle_right_idx = point_indices_sorted[middle_right_pos]
#
#         # TODO: find median point value without sorting or using mean value as partition value
#         # set root.value for the node X
#         middle_left_point_val = db[middle_left_idx, axis]
#         middle_right_point_val = db[middle_right_idx, axis]
#         root.value = (middle_left_point_val + middle_right_point_val) / 2
#
#         # build left subtree for node X
#         root.left = kdtree_recursively_build(
#             root.left, db,
#             point_indices_sorted[0: middle_right_pos],
#             update_axis(axis, db.shape[1]),
#             leaf_size
#         )
#
#         # build left subtree for node X
#         root.right = kdtree_recursively_build(
#             root.right, db,
#             point_indices_sorted[middle_right_pos:],
#             update_axis(axis, db.shape[1]),
#             leaf_size
#         )
#     return root


def kdtree_recursively_build(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        point_indices_sorted, _ = sort_keys_by_values(point_indices, db[point_indices, axis])  # M
        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1
        middle_left_point_idx = point_indices_sorted[middle_left_idx]
        middle_left_point_value = db[middle_left_point_idx, axis]

        middle_right_idx = middle_left_idx + 1
        middle_right_point_idx = point_indices_sorted[middle_right_idx]
        middle_right_point_value = db[middle_right_point_idx, axis]

        root.value = (middle_left_point_value + middle_right_point_value) * 0.5
        # === get the split position ===
        root.left = kdtree_recursively_build(root.left,
                                             db,
                                             point_indices_sorted[0:middle_right_idx],
                                             update_axis(axis, dim=db.shape[1]),
                                             leaf_size)
        root.right = kdtree_recursively_build(root.right,
                                              db,
                                              point_indices_sorted[middle_right_idx:],
                                              update_axis(axis, dim=db.shape[1]),
                                              leaf_size)
    return root


# def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
#     if root is None:
#         return
#
#     if root.is_leaf():
#         global my_count
#         my_count += 1
#         leaf_points = db[root.point_indices, :]
#         diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
#         for i in range(diff.shape[0]):
#             result_set.add_point(diff[i], root.point_indices[i])
#         return
#
#     if query[root.axis] <= root.value:
#         kdtree_knn_search(root.left, db, result_set, query)
#         if math.fabs(query[root.axis] - root.value) < result_set.get_worst_dist():
#             kdtree_knn_search(root.right, db, result_set, query)
#     else:
#         kdtree_knn_search(root.right, db, result_set, query)
#         if math.fabs(query[root.axis] - root.value) < result_set.get_worst_dist():
#             kdtree_knn_search(root.left, db, result_set, query)
#
#     # return  # TODO: no need to call return here, function ends without any other operations, and return value has no affect

def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.get_worst_dist():
            kdtree_knn_search(root.right, db, result_set, query)
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.get_worst_dist():
            kdtree_knn_search(root.left, db, result_set, query)

    return False


def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return

    if root.is_leaf():

        leaf_points = db[root.point_indices]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return

    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.get_worst_dist():
            kdtree_radius_search(root.right, db, result_set, query)
    else:
        kdtree_radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.get_worst_dist():
            kdtree_radius_search(root.left, db, result_set, query)

    # return


def tree_depth(root: Node):
    if root is None:
        return 0
    else:
        d_left = tree_depth(root.left)
        d_right = tree_depth(root.right)
        return max(d_left + 1, d_right + 1)


def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]
    print(N, dim)
    root = None
    root = kdtree_recursively_build(root,
                                    db_np,
                                    np.arange(N),
                                    axis=0,
                                    leaf_size=leaf_size)
    print("Tree depth ", tree_depth(root))
    return root


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


def read_test_file(path):
    with open(path, 'r') as f:
        points_number = int(f.readline())
        points = []
        for i in range(points_number):
            xyz = [[float(x) for x in f.readline().split()]]
            points.append(xyz)
        points = np.array(points)
        points = np.squeeze(points, axis=1)
    return points


def correctness_base():
    # points = read_velodyne_bin("000000.bin")[0:100000, :]
    points = read_test_file("../test.txt")
    point_indices = np.array(range(points.shape[0]))
    print(points.shape)
    K = 8
    query_point = points[1000]
    leaf_size = 32
    radius = 0.5

    print("******** KNN search (build with median) ***********")
    start = time.time()
    root = kdtree_construction(points, leaf_size)
    build = time.time()
    knn_result_set = KNNResultSet(K)
    kdtree_knn_search(root, points, knn_result_set, query_point)
    end = time.time()
    print("KDTree build takes {}ms".format(1000 * (build - start)))
    print("KDTree KNN search takes {}ms".format(1000 * (end - build)))
    print("Comparision times = {}".format(knn_result_set.comparision_count))
    knn_result_set.list()
    print("")

    # print("******** RNN search (build with median) ***********")
    # start = time.time()
    # root = kdtree_construction(points, leaf_size)
    # result_set_rnn = RadiusNNResultSet(radius)
    # kdtree_radius_search(root, points, result_set_rnn, np.asarray(query_point))
    # result_set_rnn.list()
    # print("Total time {}ms".format(1000 * (time.time() - start)))
    # print("Comparison times: ", result_set_rnn.comparision_count)
    # print("")

    # print("******** KNN search (build with mean) ***********")
    # start = time.time()
    # root = None
    # root = kdtree_recursively_build_fast(root, points, point_indices, 0, leaf_size)
    # build = time.time()
    # knn_result_set = KNNResultSet(K)
    # kdtree_knn_search(root, points, knn_result_set, query_point)
    # end = time.time()
    # print("KDTree build takes {}ms".format(1000 * (build - start)))
    # print("KDTree KNN search takes {}ms".format(1000 * (end - build)))
    # print("Comparision times = {}".format(knn_result_set.comparision_count))
    # knn_result_set.list()
    # print("")

    # print("******** Open3d ***********")
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # start = time.time()
    # pcd_tree_o3d = o3d.geometry.KDTreeFlann(pcd)
    # build = time.time()
    # _, idx, dist = pcd_tree_o3d.search_knn_vector_3d(query_point, K)
    # end = time.time()
    # print("Open3d build time {}ms".format(1000 * (build - start)))
    # print("Open3d search time {}ms".format(1000 * (end - build)))
    # print("Open3d total time {}ms".format(1000 * (end - start)))
    # print(idx)
    # print(dist)


if __name__ == "__main__":
    correctness_base()
