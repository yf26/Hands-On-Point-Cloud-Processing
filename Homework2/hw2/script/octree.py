import struct
import time

import numpy as np
from script.result_set import KNNResultSet, RadiusNNResultSet
import open3d as o3d


class Octant:
    def __init__(self, children, center: np.ndarray, extent, point_indices: list, is_leaf):
        self.children = children
        self.center = center
        self.extent = extent
        self.point_indices = point_indices
        self.is_leaf = is_leaf


def octree_build_recursively(root: Octant, db, center: np.ndarray, extent, point_indices: list, leaf_size, min_extent):
    if len(point_indices) == 0:
        return None

    if root is None:
        root = Octant([None for i in range(8)], center, extent, point_indices, is_leaf=True)

    if len(point_indices) <= leaf_size or extent <= min_extent:
        root.is_leaf = True
    else:
        root.is_leaf = False
        children_point_indices = [[] for i in range(8)]
        for point_idx in point_indices:
            morton_code = 0
            if db[point_idx][0] > center[0]:
                morton_code = morton_code | 1
            if db[point_idx][1] > center[1]:
                morton_code = morton_code | 2
            if db[point_idx][2] > center[2]:
                morton_code = morton_code | 4
            children_point_indices[morton_code].append(point_idx)

        factor = [-0.5, 0.5]
        for i in range(8):
            child_center_x = center[0] + factor[(i & 1) > 0] * extent
            child_center_y = center[1] + factor[(i & 2) > 0] * extent
            child_center_z = center[2] + factor[(i & 4) > 0] * extent
            child_extent = 0.5 * extent
            child_center = np.asarray([child_center_x, child_center_y, child_center_z])  # TODO
            root.children[i] = octree_build_recursively(
                root.children[i], db, child_center, child_extent,
                children_point_indices[i], leaf_size, min_extent
            )
    return root


def inside(query: np.ndarray, radius: float, octant: Octant):
    """
    determine if the query ball is inside the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    possible_space = query_offset_abs + radius
    return np.all(possible_space < octant.extent)


def overlaps(query: np.ndarray, radius: float, octant: Octant):
    """
    determines if the query ball overlaps with the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    # completely outside
    max_dist = radius + octant.extent
    if np.any(query_offset_abs > max_dist):
        return False

    # check if the query ball contacts one face of the octant
    if np.sum((query_offset_abs < octant.extent).astype(np.int)) >= 2:
        return True

    # check if the query ball contacts one edge or one corner
    x_diff = max(query_offset_abs[0] - octant.extent, 0)
    y_diff = max(query_offset_abs[1] - octant.extent, 0)
    z_diff = max(query_offset_abs[2] - octant.extent, 0)

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius


def contains(query: np.ndarray, radius: float, octant: Octant):
    """
    determines if query ball contains the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    return np.linalg.norm(query_offset_abs + octant.extent) < radius


def octree_knn_search(root: Octant, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    # traverse the leaf points
    if root.is_leaf and len(root.point_indices) > 0:
        leaf_points = db[root.point_indices]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return inside(query, result_set.get_worst_dist(), root)

    # root is not leaf, find the most relevant child octant
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4

    if octree_knn_search(root.children[morton_code], db, result_set, query):
        return True

    # check other children
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if not overlaps(query, result_set.get_worst_dist(), child):  # TODO
            continue
        if octree_knn_search(child, db, result_set, query):
            return True

    # final check if we can stop search
    return inside(query, result_set.get_worst_dist(), root)


def octree_radius_search(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.get_worst_dist(), root)

    # go to the relevant child first
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4

    if octree_radius_search(root.children[morton_code], db, result_set, query):
        return True

    # check other children
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if False == overlaps(query, result_set.get_worst_dist(), child):
            continue
        if octree_radius_search(child, db, result_set, query):
            return True

    # final check of if we can stop search
    return inside(query, result_set.get_worst_dist(), root)


def octree_construction(db: np.ndarray, leaf_size, min_extent):
    N, dim = db.shape[0], db.shape[1]
    db_min = np.amin(db, axis=0)
    db_max = np.amax(db, axis=0)
    db_extent = np.max(db_max - db_min) * 0.5
    db_center = db_min + db_extent

    root = None
    root = octree_build_recursively(root, db, db_center, db_extent, list(range(N)),
                                    leaf_size, min_extent)
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


def correctness_base():
    points = read_velodyne_bin("000000.bin")[0:100000, :]
    K = 8
    query_point = points[1000]
    leaf_size = 32
    min_extent = 0.0001

    print("******** KNN search based on Octree ***********")
    start = time.time()
    root = octree_construction(points, leaf_size, min_extent)
    build = time.time()
    knn_result_set = KNNResultSet(K)
    octree_knn_search(root, points, knn_result_set, query_point)
    end = time.time()
    print("Octree build takes {}ms".format(1000 * (build - start)))
    print("Octree KNN search takes {}ms".format(1000 * (end - build)))
    print("Comparision times = {}".format(knn_result_set.comparision_count))
    knn_result_set.list()
    print("")

    print("******** RNN search based on Octree ***********")
    start = time.time()
    root = octree_construction(points, leaf_size, min_extent)
    result_set_rnn = RadiusNNResultSet(0.5)
    octree_radius_search(root, points, result_set_rnn, np.asarray(query_point))
    result_set_rnn.list()
    print("Total time {}ms".format(1000 * (time.time() - start)))
    print("Comparison times: ", result_set_rnn.comparision_count)
    print("")

    print("******** Open3d ***********")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    start = time.time()
    pcd_tree_o3d = o3d.geometry.KDTreeFlann(pcd)
    build = time.time()
    _, idx, dist = pcd_tree_o3d.search_knn_vector_3d(query_point, K)
    end = time.time()
    print("Open3d build time {}ms".format(1000 * (build - start)))
    print("Open3d search time {}ms".format(1000 * (end - build)))
    print("Open3d total time {}ms".format(1000 * (end - start)))
    print(idx)
    print(dist)


if __name__ == "__main__":
    correctness_base()