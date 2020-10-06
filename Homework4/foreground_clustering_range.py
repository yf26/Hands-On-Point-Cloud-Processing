import time
import numpy as np
import math
from collections import deque

from sklearn.cluster import spectral_clustering
import open3d as o3d
import matplotlib.pyplot as plt

from ground_detection_SVD import ground_detection_on3segs, plot_clusters, read_velodyne_bin, pcd_preprocessing


def pcd_to_range_image(pcd_points, resolution):
    """
    :param pcd_points: Nx3 np.array
    :param resolution: degree
    :return:
    """
    resolution_rad = math.pi / 180 * resolution
    N = pcd_points.shape[0]
    width = math.floor(360 / resolution) + 1
    height = math.floor(60 / resolution) + 1

    offset_width = math.ceil(width / 2)
    offset_height = math.ceil(height / 2)

    range_image = np.full((height, width), -1, dtype=float)
    # store every pcd index belonging to the same range image pixel
    idx_image = np.empty((height, width), dtype=object)

    d = np.linalg.norm(pcd_points, axis=1)
    for i in range(N):
        alpha = math.atan2(pcd_points[i, 1], pcd_points[i, 0])  # horizontal
        beta = math.asin(pcd_points[i, 2], math.sqrt(
            pcd_points[i, 0] * pcd_points[i, 0] + pcd_points[i, 1] * pcd_points[i, 1]))  # vertical
        x = width - 1 - (math.floor(alpha / resolution_rad) + offset_width)
        y = height - 1 - (math.floor(beta / resolution_rad) + offset_height)
        range_image[y, x] = d[i]
        idx_image[y, x] = np.append(idx_image[y, x], i)

    row_filter = np.logical_not(np.all(range_image == -1, axis=1))
    # row_range = np.array(range(row_filter.shape[0]))
    # row_start = row_range[row_filter][0]
    range_image = range_image[row_filter]
    idx_image = idx_image[row_filter]
    col_filter = np.logical_not(np.all(range_image == -1, axis=0))
    return range_image[:, col_filter], idx_image[:, col_filter], d
    # return range_image, idx_image


def range_image_labeling(range_image, idx_image, depth_list, phi, theta, nn_mode):
    phi = phi * math.pi / 180
    threshold = theta * math.pi / 180
    label = 0
    [rows, cols] = range_image.shape
    image_label = np.full((rows, cols), -1, dtype=int)
    for r in range(rows):
        for c in range(cols):
            # print("processing {}, {}".format(r, c))
            if image_label[r, c] == -1 and range_image[r, c] > 0:
                queue = deque()
                queue.append([r, c])
                image_label[r, c] = label

                while len(queue) != 0:
                    [r, c] = queue[0]
                    queue.popleft()
                    # image_label[r, c] = label
                    # for [rn, cn] in neighbourhood(r, c, mode=nn_mode):
                    for rn in range(r-nn_mode, r+nn_mode+1):
                        for cn in range(c-nn_mode, c+nn_mode+1):
                            if rn < 0 or rn > rows - 1:
                                continue
                            if cn < 0:
                                cn += cols
                            if cn >= cols:
                                cn -= cols
                            if image_label[rn, cn] != -1:
                                continue
                            d1 = max(range_image[r, c], range_image[rn, cn])
                            d2 = min(range_image[r, c], range_image[rn, cn])
                            # d_rc = get_dist(r, c, idx_image, depth_list)
                            # d_rncn = get_dist(rn, cn, idx_image, depth_list)
                            # d1 = max(d_rc, d_rncn)
                            # d2 = min(d_rc, d_rncn)
                            if d1 == -1 or d2 == -1:
                                continue
                            angle_to_test = math.atan2(d2 * math.sin(phi), (d1 - d2 * math.cos(phi)))
                            if angle_to_test > threshold and math.fabs(d1 - d2) < 1 and range_image[rn, cn] > 0:
                                # if math.fabs(d1 - d2) < 0.4 and image_label[rn, cn] == -1 and range_image[rn, cn] > 0:
                                queue.append([rn, cn])
                                image_label[rn, cn] = label

                label += 1
    return image_label


def get_dist(r, c, idx_image, depth_list):
    if idx_image[r, c] is None:
        return -1
    idx_at_pixel = np.array(idx_image[r, c][1:], dtype=int)
    return np.min(depth_list[idx_at_pixel])


def neighbourhood(r, c, mode):
    if mode == 4:
        return [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]]
    if mode == 8:
        return [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1],
                [r - 1, c - 1], [r + 1, c - 1], [r - 1, c + 1], [r + 1, c + 1]]
    if mode == 12:
        return [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1],
                [r - 1, c - 1], [r + 1, c - 1], [r - 1, c + 1], [r + 1, c + 1],
                [r, c - 2], [r, c + 2], [r - 2, c], [r + 2, c]]
    if mode == 20:
        return [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1],
                [r - 1, c - 1], [r + 1, c - 1], [r - 1, c + 1], [r + 1, c + 1],
                [r, c - 2], [r, c + 2], [r - 2, c], [r + 2, c],
                [r - 2, c - 1], [r - 2, c + 1], [r + 2, c - 1], [r + 2, c + 1],
                [r - 1, c - 2], [r + 1, c - 2], [r - 1, c + 2], [r + 1, c + 2]]



def cluster_assignment(idx_image, image_label, pcd_size):
    assert (idx_image.shape == image_label.shape)
    cluster_idx = np.full(pcd_size, -1, dtype=int)
    [rows, cols] = image_label.shape
    for r in range(rows):
        for c in range(cols):
            if idx_image[r, c] is not None:
                indices = np.array(idx_image[r, c][1:], dtype=int)
                cluster_idx[indices] = image_label[r, c]
    return cluster_idx


def depth_completion(image, pad):
    [rows, cols] = image.shape
    result_dila = np.full_like(image, -1, dtype=float)
    result_closing = np.full_like(image, -1, dtype=float)
    # dilation
    for r in range(pad, rows - pad):
        for c in range(pad, cols - pad):
            result_dila[r, c] = np.amax(image[r - pad: r + pad + 1, c - pad: c + pad + 1])
    # erosion
    for r in range(pad, rows - pad):
        for c in range(pad, cols - pad):
            result_closing[r, c] = np.amin(result_dila[r - pad: r + pad + 1, c - pad: c + pad + 1])

    return result_closing


if __name__ == "__main__":
    path = "test/000099.bin"
    points = read_velodyne_bin(path)
    points = pcd_preprocessing(points)

    start = time.time()
    ground_idx, foreground_idx = ground_detection_on3segs(points)
    print("Ground detection takes {}ms".format(1000 * (time.time() - start)))

    # convert to range image, cluster on range image
    start = time.time()
    resolution = 0.7
    range_image, idx_image, depth_list = pcd_to_range_image(points[foreground_idx], resolution)
    # connected_range_image = depth_completion(range_image, 1)
    image_label = range_image_labeling(range_image, idx_image, depth_list, resolution, 30, nn_mode=7)
    cluster_idx = cluster_assignment(idx_image, image_label, foreground_idx.shape[0])
    print("Clustering takes {}ms".format(1000 * (time.time() - start)))

    plt.figure(figsize=(12, 4))
    plt.subplot(311)
    plt.axis('off')
    plt.imshow(range_image)
    plt.subplot(312)
    plt.axis('off')
    plt.imshow(range_image)
    plt.subplot(313)
    plt.axis('off')
    plt.imshow(image_label)
    plt.show()

    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(points[ground_idx])
    ground_pcd.paint_uniform_color([0.8, 0.8, 0.8])
    foreground_pcd = o3d.geometry.PointCloud()
    foreground_pcd.points = o3d.utility.Vector3dVector(points[foreground_idx])
    plot_clusters(ground_pcd, foreground_pcd, np.asarray(cluster_idx))
