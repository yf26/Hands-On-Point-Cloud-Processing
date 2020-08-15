import struct
import time
from itertools import islice, cycle
import numpy as np
import open3d as o3d
import bottleneck as bn
import mylib


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


def pcd_preprocessing(data):
    y_left = 30
    y_right = -15
    y_range_filter = np.logical_and(data[:, 1] < y_left, data[:, 1] > y_right)
    data = data[y_range_filter]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    # o3d.visualization.draw_geometries([pcd])
    pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.7)
    # o3d.visualization.draw_geometries([pcd_filtered])
    return np.asarray(pcd_filtered.points)


def visualize_pcd(points, rgb=None):
    pcd_seg = o3d.geometry.PointCloud()
    pcd_seg.points = o3d.utility.Vector3dVector(points)
    if rgb is not None:
        pcd_seg.paint_uniform_color(rgb)
    o3d.visualization.draw_geometries([pcd_seg])


def extract_initial_seeds(pcd_points, LPR_size, threshold_seeds):
    z_high = -1.73 + 0.5
    # z_low = -1.73 - 0.5
    # z_filter = np.logical_and(pcd_points[:, 2] < z_high, pcd_points[:, 2] > z_low)
    z_filter = pcd_points[:, 2] < z_high

    possible_ground_points = pcd_points[z_filter, :]

    if LPR_size > possible_ground_points.shape[0]:
        # print("LPR_size is too large!")
        LPR_idx = np.array(range(possible_ground_points.shape[0]))
        # print(LPR_idx)
    else:
        LPR_idx = bn.argpartition(possible_ground_points[:, 2], LPR_size - 1)[:LPR_size]
        assert (LPR_idx.shape[0] == LPR_size)

    LPR = np.mean(possible_ground_points[LPR_idx, :], axis=0)
    # print(possible_ground_points[LPR_idx, :])
    # print(LPR)
    upper_bound = LPR[2] + threshold_seeds

    seeds_filter = possible_ground_points[:, 2] < upper_bound
    seeds = possible_ground_points[seeds_filter, :]
    return seeds


def estimate_plane(pcd_points):
    center_point = np.mean(pcd_points, axis=0)
    centered_data = np.subtract(pcd_points, center_point)
    XTX = centered_data.transpose().dot(centered_data)
    # eigenvalues, eigenvectors = np.linalg.eig(XTX)
    # sort = eigenvalues.argsort()[::-1]
    # eigenvalues = eigenvalues[sort]
    # eigenvectors = eigenvectors[:, sort]
    normal = mylib.FastEigen3x3(XTX)
    d = - normal.dot(center_point)
    return np.r_[normal, d]


def ground_detection(pcd_points, pcd_indices, max_iter, LPR_size, threshold_dist):
    pcd_size = pcd_points.shape[0]
    seeds = extract_initial_seeds(pcd_points, LPR_size, threshold_dist)

    inliers_filter = None
    for i in range(max_iter):
        params = estimate_plane(seeds)
        # calculate inliers
        dists = np.fabs(np.c_[pcd_points, np.ones((pcd_size, 1))].dot(params))
        inliers_filter = dists < threshold_dist
        seeds = pcd_points[inliers_filter]
    foreground_filter = np.logical_not(inliers_filter)
    print(params)
    return seeds, pcd_indices[inliers_filter], pcd_indices[foreground_filter]


def ground_detection_on3segs(pcd_points, main_dist=20, max_iter=6, threshold_dist=0.18):
    x_min = np.min(pcd_points[:, 0])
    x_max = np.max(pcd_points[:, 0])
    segments_x = [x_min, -main_dist, main_dist, x_max]
    # print(segments_x)
    segments_size = len(segments_x) - 1

    total_indices = np.array(range(pcd_points.shape[0]))

    stacked_ground_idx = np.empty(0, dtype=int)
    stacked_foregr_idx = np.empty(0, dtype=int)
    for i in range(segments_size):
        range_filter = np.logical_and(pcd_points[:, 0] < segments_x[i+1], pcd_points[:, 0] > segments_x[i])
        # print(segments_x[i], segments_x[i+1])
        segmented_points = pcd_points[range_filter]
        segmented_indices = total_indices[range_filter]
        # visualize_pcd(segmented_points)
        _, ground_filter, foreground_filter = ground_detection(segmented_points, segmented_indices, max_iter, LPR_size=10000, threshold_dist=threshold_dist)
        stacked_ground_idx = np.r_[stacked_ground_idx, total_indices[ground_filter]]
        stacked_foregr_idx = np.r_[stacked_foregr_idx, total_indices[foreground_filter]]
    return stacked_ground_idx, stacked_foregr_idx


def hex_to_rgb(color_list):
    rgb = []
    for h in color_list:
        h = h.lstrip("#")
        rgb.append([int(h[i:i + 2], 16)/255 for i in (0, 2, 4)])
    return rgb


def plot_clusters(g_pcd, fg_pcd, cluster_index):
    hex = ['#377eb8', '#ff7f00', '#4daf4a',
           '#f781bf', '#a65628', '#984ea3',
           '#999999', '#e41a1c', '#dede00']

    colors = np.array(list(islice(cycle(hex),
                                  int(max(cluster_index) + 1))))
    colors = np.append(colors, "#000000")
    color_map = hex_to_rgb(colors)

    assert isinstance(g_pcd, o3d.geometry.PointCloud)
    assert isinstance(fg_pcd, o3d.geometry.PointCloud)

    fg_colors = []
    for c in cluster_index:
        fg_colors.append(color_map[c])
    fg_pcd.colors = o3d.utility.Vector3dVector(fg_colors)
    o3d.visualization.draw_geometries([g_pcd, fg_pcd])


if __name__ == "__main__":
    start = time.time()
    path = "test/000111.bin"
    points = read_velodyne_bin(path)
    points = pcd_preprocessing(points)
    print("Preprocessing takes {}ms".format(1000 * (time.time() - start)))

    start = time.time()
    ground_idx, foreground_idx = ground_detection_on3segs(points)
    # _, ground_idx = ground_detection(points, np.array(range(points.shape[0])), 10, 10000, 0.15)
    print("Ground detection takes {}ms".format(1000 * (time.time() - start)))

    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(points[ground_idx])
    ground_pcd.paint_uniform_color([0.8, 0.8, 0.8])

    foreground_pcd = o3d.geometry.PointCloud()
    foreground_pcd.points = o3d.utility.Vector3dVector(points[foreground_idx])

    # TODO my clustering
    start = time.time()
    clusters = foreground_pcd.cluster_dbscan(0.8, 20, print_progress=True)
    print("Clustering takes {}ms".format(1000 * (time.time() - start)))

    plot_clusters(ground_pcd, foreground_pcd, np.asarray(clusters))