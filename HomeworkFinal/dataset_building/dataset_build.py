import math
import os
from array import array
from itertools import islice, cycle

from tqdm import tqdm
import struct
import time
import linecache
import numpy as np
import open3d as o3d

from dataset_building.ground_detection_SVD import pcd_preprocessing, ground_detection_on3segs


def raw_kitti_bin_reader(path, file_name):
    return np.fromfile(path + file_name + ".bin", dtype=np.float32, count=-1).reshape(-1, 4)


def view_single_pcd_from_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def visualize_pcd_with_label(points, objects_idx_dict, bbox_points):
    pcd_list = []
    foreground_idx = []
    colors = {"Car": [1, 0, 1], "Pedestrian": [1, 0, 0], "Cyclist": [0, 0, 1]}
    for cls in objects_idx_dict:
        idx_concat = [idx for idx_sublist in objects_idx_dict[cls] for idx in idx_sublist]
        foreground_idx.append(idx_concat)
        if len(idx_concat) == 0: continue
        pcd_cls = o3d.geometry.PointCloud()
        pcd_cls.points = o3d.utility.Vector3dVector(points[idx_concat, :3])
        pcd_cls.paint_uniform_color(colors[cls])
        pcd_list.append(pcd_cls)

    foreground_idx = [idx for idx_sublist in foreground_idx for idx in idx_sublist]
    background_idx = np.delete(range(points.shape[0]), foreground_idx)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[background_idx, :3])
    pcd.paint_uniform_color([0.8, 0.8, 0.8])
    pcd_list.append(pcd)

    if bbox_points is not None:
        bbox_pcd = o3d.geometry.PointCloud()
        bbox_pcd.points = o3d.utility.Vector3dVector(bbox_points)
        bbox_pcd.paint_uniform_color([0, 0, 0])
        pcd_list.append(bbox_pcd)
    o3d.visualization.draw_geometries(pcd_list)


def bbox_reader(label_path, calib_path, file_name):
    Tr = linecache.getline(calib_path + file_name + ".txt", 6).split()[1:]
    Tr = np.array(Tr, dtype=np.float32).reshape(3, 4)
    R_inv = Tr[:, 0:3].transpose()
    t_inv = - R_inv.dot(Tr[:, 3])

    class_name = ["Car", "Pedestrian", "Cyclist"]
    bbox_dict = {cls: [] for cls in class_name}
    with open(label_path + file_name + ".txt", "r") as f:
        while True:
            line = f.readline().split()
            if len(line) == 0: break
            if line[0] in class_name:
                hwl = np.array(line[8:11], dtype=np.float32)
                theta_y = float(line[-1])
                bbox_center_camera = np.array(line[-4:-1], dtype=np.float32)
                bbox_front_camera = bbox_center_camera + [hwl[2] / 2 * np.cos(-theta_y), 0,
                                                          hwl[2] / 2 * np.sin(-theta_y)]
                bbox_center = R_inv.dot(bbox_center_camera) + t_inv
                bbox_front = R_inv.dot(bbox_front_camera) + t_inv
                bbox_dict[line[0]].append((bbox_center, bbox_front, hwl))
    return bbox_dict


def get_objects_idx_dict(points, bbox_dict, get_8_pts=False):
    class_name = ["Car", "Pedestrian", "Cyclist"]
    points_idx_dict = {cls: [] for cls in class_name}
    if get_8_pts:
        bbox_points = np.empty((0, 3))
        for cls in bbox_dict:
            for bbox_params in bbox_dict[cls]:
                idx, bbox_8_pts = get_points_idx_in_bbox(points, bbox_params, get_8_pts)
                bbox_points = np.append(bbox_points, bbox_8_pts, axis=0)
                points_idx_dict[cls].append(idx)
        return points_idx_dict, bbox_points
    else:
        for cls in bbox_dict:
            for bbox_params in bbox_dict[cls]:
                idx, _ = get_points_idx_in_bbox(points, bbox_params, get_8_pts)
                points_idx_dict[cls].append(idx)
        return points_idx_dict, None


def get_points_idx_in_bbox(points, bbox_params, get_8_pts=False):
    bbox_center, bbox_front, hwl = bbox_params

    orient_vec = bbox_front - bbox_center
    bbox_8_pts = None
    if get_8_pts:
        '''rectangle vertexes'''
        orient_vec_normal = np.array([orient_vec[1], -orient_vec[0], 0])
        orient_vec_normal = hwl[1] / (2 * np.linalg.norm(orient_vec_normal)) * orient_vec_normal

        bbox_8_pts = np.array([bbox_front + orient_vec_normal,
                               bbox_front - orient_vec_normal,
                               2 * bbox_center - bbox_front + orient_vec_normal,
                               2 * bbox_center - bbox_front - orient_vec_normal])
        bbox_8_pts = np.vstack((bbox_8_pts, bbox_8_pts + [0, 0, hwl[0]]))

    '''params for bbox'''
    h, w, l = hwl[0], hwl[1], hwl[2]
    center = bbox_center + [0, 0, h / 2]
    theta_z_lidar = math.atan2(orient_vec[1], orient_vec[0])
    R = np.array([[np.cos(theta_z_lidar), - np.sin(theta_z_lidar), 0],
                  [np.sin(theta_z_lidar), np.cos(theta_z_lidar), 0],
                  [0, 0, 1]])
    extent = [1.15 * l, 1.25 * w, h]

    '''bbox filter'''
    bbox_o3d = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    indices = bbox_o3d.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points))
    return indices, bbox_8_pts


def hex_to_rgb(color_list):
    rgb = []
    for h in color_list:
        h = h.lstrip("#")
        rgb.append([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return rgb


def plot_clusters(pcd, cluster_index):
    hex = ['#377eb8', '#ff7f00', '#4daf4a',
           '#f781bf', '#a65628', '#984ea3',
           '#999999', '#e41a1c', '#dede00']

    colors = np.array(list(islice(cycle(hex),
                                  int(max(cluster_index) + 1))))
    colors = np.append(colors, "#000000")
    color_map = hex_to_rgb(colors)

    assert isinstance(pcd, o3d.geometry.PointCloud)

    fg_colors = []
    for c in cluster_index:
        fg_colors.append(color_map[c])
    pcd.colors = o3d.utility.Vector3dVector(fg_colors)
    o3d.visualization.draw_geometries([pcd])


def build(pcd_path, label_path, calib_path, my_db_path):

    if not os.path.exists(my_db_path):
        os.mkdir(my_db_path)
        os.mkdir(os.path.join(my_db_path, "Car"))
        os.mkdir(os.path.join(my_db_path, "Cyclist"))
        os.mkdir(os.path.join(my_db_path, "Pedestrian"))
        os.mkdir(os.path.join(my_db_path, "DontCare"))
    else:
        print("{} already exists!".format(my_db_path))
        return

    file_name_list = os.listdir(pcd_path)

    for file_name in tqdm(file_name_list, total=len(file_name_list), smoothing=0.9):
        file_name = file_name.rstrip('.bin')
        # print(file_name)
        points = raw_kitti_bin_reader(pcd_path, file_name)

        bbox_dict = bbox_reader(label_path, calib_path, file_name)
        objects_idx_dict, bbox_pts = get_objects_idx_dict(points[:, :3], bbox_dict, get_8_pts=False)

        train_obj_idx = []

        # print(objects_idx_dict)
        # for cls in objects_idx_dict:
        #     print(cls, len(objects_idx_dict[cls]))

        # store the instances of Car, Pedestrian and Cyclist
        for cls in objects_idx_dict:
            # concatenate all indices of the instances from same class
            idx_concat = [idx for idx_sublist in objects_idx_dict[cls] for idx in idx_sublist]
            train_obj_idx.append(idx_concat)
            for num, idx_sublist in enumerate(objects_idx_dict[cls]):
                if len(idx_sublist) < 20:
                    continue
                instance_name = cls + "_" + file_name + "_" + str(num) + '.bin'
                instance_points = points[idx_sublist, :3]
                instance_points = instance_points.flatten()
                output_file = open(os.path.join(my_db_path, cls, instance_name), 'wb')
                float_array = array('f', instance_points)
                float_array.tofile(output_file)
                output_file.close()

        # visualize the labeled train objects
        # visualize_pcd_with_label(points, objects_idx_dict, bbox_pts)

        # delete labeled train objects points from total points
        train_obj_idx = [idx for idx_sublist in train_obj_idx for idx in idx_sublist]
        other_obj_idx = np.delete(range(points.shape[0]), train_obj_idx)

        # ground removing and preparing for clustering on other objs
        remain_points = pcd_preprocessing(points[other_obj_idx, :3], match_image_range=True)
        try:
            _, fg_idx = ground_detection_on3segs(remain_points, match_image_range=True)
        except IndexError:
            continue

        remain_points = remain_points[fg_idx]
        remain_pcd = o3d.geometry.PointCloud()
        remain_pcd.points = o3d.utility.Vector3dVector(remain_points)
        cluster_results = remain_pcd.cluster_dbscan(0.8, 20, print_progress=False)

        # visualize the clustered other objects
        # plot_clusters(remain_pcd, cluster_results)
        cluster_idx_dict = {}
        for idx, cluster in enumerate(cluster_results):
            try:
                cluster_idx_dict[cluster].append(idx)
            except KeyError:
                cluster_idx_dict[cluster] = [idx]

        # store the instances of DontCare
        num = 0
        for cls in cluster_idx_dict:
            if cls == -1: continue
            if cls % 8 == 0:  # only store the 12.5% of DontCare objects
                indices = cluster_idx_dict[cls]
                if len(indices) > 50:
                    instance_name = "DontCare_" + file_name + "_" + str(num) + '.bin'
                    num += 1
                    instance_points = remain_points[indices, :]
                    instance_points = instance_points.flatten()
                    output_file = open(os.path.join(my_db_path, "DontCare", instance_name), 'wb')
                    float_array = array('f', instance_points)
                    float_array.tofile(output_file)
                    output_file.close()
    # visualize_pcd_with_label(points, objects_idx_dict, bbox_pts)


def generate_train_test_split_file(my_db_path):
    train_data_rate = 0.7

    car_files = os.listdir(os.path.join(my_db_path, "Car"))
    cyclist_files = os.listdir(os.path.join(my_db_path, "Cyclist"))
    pedestrian_files = os.listdir(os.path.join(my_db_path, "Pedestrian"))
    dontcare_files = os.listdir(os.path.join(my_db_path, "DontCare"))

    all_files = [car_files, cyclist_files, pedestrian_files, dontcare_files]

    train_path = "../dataset_info/mykitti_train.txt"
    test_path = "../dataset_info/mykitti_test.txt"

    if os.path.exists(test_path) and os.path.exists(train_path):
        print("Split files already exist!")
        return

    train_f = open(train_path, "w")
    test_f = open(test_path, "w")

    for files in all_files:
        idx_split = math.ceil(train_data_rate * len(files))
        for fn in files[0: idx_split]:
            train_f.write(fn+'\n')
        for fn in files[idx_split: -1]:
            test_f.write(fn + '\n')

    train_f.close()
    test_f.close()
    print("Split train test files done!")


def data_augmentation(my_db_path):
    car_files = os.listdir(os.path.join(my_db_path, "Car"))
    cyclist_files = os.listdir(os.path.join(my_db_path, "Cyclist"))
    pedestrian_files = os.listdir(os.path.join(my_db_path, "Pedestrian"))
    dontcare_files = os.listdir(os.path.join(my_db_path, "DontCare"))

    def random_jitter(points):
        points += np.random.normal(0, 0.01, size=points.shape)  # random jitter
        return points

    def random_scale(points):
        scale_low = 0.8
        scale_high = 1.25
        points = points * np.random.uniform(scale_low, scale_high)
        return points

    def random_rotate_local_z(points):
        center = np.mean(points, axis=0)
        centered_points = points - np.expand_dims(center, 0)
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        rotated_points = rotation_matrix.dot(centered_points.transpose())
        points = rotated_points.transpose() + np.expand_dims(center, 0)
        return points

    def random_dropout(points):
        dropout_ratio = np.random.random() * 0.5  # random dropout points
        drop_idx = np.where(np.random.random(points.shape[0]) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            points[drop_idx, :] = points[0, :]  # set to first point
        return points

    # processing cyclist
    print("Data augmentation for class Cyclist:")
    cyclist_path = os.path.join(my_db_path, "Cyclist")
    for file in tqdm(cyclist_files, total=len(cyclist_files), smoothing=0.9):
        file_path = os.path.join(cyclist_path, file)
        points_raw = np.fromfile(file_path, dtype=np.float32, count=-1).reshape(-1, 3)

        # Augmentation 1: rotation
        points = random_rotate_local_z(points_raw)
        instance_name = file.rstrip(".bin") + "r" + ".bin"
        instance_points = points.flatten()
        output_file = open(os.path.join(cyclist_path, instance_name), 'wb')
        float_array = array('f', instance_points)
        float_array.tofile(output_file)
        output_file.close()

        # Augmentation 2: rotation + dropout
        points = random_rotate_local_z(points_raw)
        points = random_dropout(points)
        instance_name = file.rstrip(".bin") + "rd" + ".bin"
        instance_points = points.flatten()
        output_file = open(os.path.join(cyclist_path, instance_name), 'wb')
        float_array = array('f', instance_points)
        float_array.tofile(output_file)
        output_file.close()

        # Augmentation 3: rotation + jitter
        points = random_rotate_local_z(points_raw)
        points = random_jitter(points)
        instance_name = file.rstrip(".bin") + "rj" + ".bin"
        instance_points = points.flatten()
        output_file = open(os.path.join(cyclist_path, instance_name), 'wb')
        float_array = array('f', instance_points)
        float_array.tofile(output_file)
        output_file.close()

        # Augmentation 4: rotation + scale
        points = random_rotate_local_z(points_raw)
        points = random_scale(points)
        instance_name = file.rstrip(".bin") + "rs" + ".bin"
        instance_points = points.flatten()
        output_file = open(os.path.join(cyclist_path, instance_name), 'wb')
        float_array = array('f', instance_points)
        float_array.tofile(output_file)
        output_file.close()

        # Augmentation 5: rotation + scale + jitter
        points = random_rotate_local_z(points_raw)
        points = random_scale(points)
        points = random_jitter(points)
        instance_name = file.rstrip(".bin") + "rsj" + ".bin"
        instance_points = points.flatten()
        output_file = open(os.path.join(cyclist_path, instance_name), 'wb')
        float_array = array('f', instance_points)
        float_array.tofile(output_file)
        output_file.close()

        # Augmentation 6: rotation + scale + dropout
        points = random_rotate_local_z(points_raw)
        points = random_scale(points)
        instance_name = file.rstrip(".bin") + "rsd" + ".bin"
        instance_points = points.flatten()
        output_file = open(os.path.join(cyclist_path, instance_name), 'wb')
        float_array = array('f', instance_points)
        float_array.tofile(output_file)
        output_file.close()

        # Augmentation 7: rotation + scale + jitter + dropout
        points = random_rotate_local_z(points_raw)
        points = random_scale(points)
        points = random_jitter(points)
        points = random_dropout(points)
        instance_name = file.rstrip(".bin") + "rsjd" + ".bin"
        instance_points = points.flatten()
        output_file = open(os.path.join(cyclist_path, instance_name), 'wb')
        float_array = array('f', instance_points)
        float_array.tofile(output_file)
        output_file.close()

    # processing pedestrian
    print("Data augmentation for class Pedestrian:")
    pedestrian_path = os.path.join(my_db_path, "Pedestrian")
    for file in tqdm(pedestrian_files, total=len(pedestrian_files), smoothing=0.9):
        file_path = os.path.join(pedestrian_path, file)
        points_raw = np.fromfile(file_path, dtype=np.float32, count=-1).reshape(-1, 3)

        # Augmentation 1: rotation
        points = random_rotate_local_z(points_raw)
        instance_name = file.rstrip(".bin") + "r" + ".bin"
        instance_points = points.flatten()
        output_file = open(os.path.join(pedestrian_path, instance_name), 'wb')
        float_array = array('f', instance_points)
        float_array.tofile(output_file)
        output_file.close()

        # Augmentation 2: rotation + scale
        points = random_rotate_local_z(points_raw)
        points = random_scale(points)
        instance_name = file.rstrip(".bin") + "rs" + ".bin"
        instance_points = points.flatten()
        output_file = open(os.path.join(pedestrian_path, instance_name), 'wb')
        float_array = array('f', instance_points)
        float_array.tofile(output_file)
        output_file.close()


def get_mean_hwl(pcd_path, label_path, calib_path):

    file_name_list = os.listdir(pcd_path)

    car_hwl = np.zeros(3)
    cyclist_hwl = np.zeros(3)
    pedestrian_hwl = np.zeros(3)
    car_num = 0
    cyclist_num = 0
    pedestrian_num = 0

    class_name = ["Car", "Cyclist", "Pedestrian"]
    class_hwl = [car_hwl, cyclist_hwl, pedestrian_hwl]
    class_num = [car_num, cyclist_num, pedestrian_num]

    for file_name in tqdm(file_name_list, total=len(file_name_list), smoothing=0.9):
        file_name = file_name.rstrip('.bin')

        bbox_dict = bbox_reader(label_path, calib_path, file_name)
        for i in range(3):
            cls = class_name[i]
            class_num[i] += len(bbox_dict[cls])
            for triple in bbox_dict[cls]:
                instance_hwl = triple[2]
                class_hwl[i] += instance_hwl

    class_hwl[0] /= class_num[0]
    class_hwl[1] /= class_num[1]
    class_hwl[2] /= class_num[2]

    return class_hwl


if __name__ == "__main__":
    pcd_path = "/disk/ml/datasets/KITTI/object/data/training/velodyne/"
    label_path = "/disk/ml/datasets/KITTI/object/data/training/label_2/"
    calib_path = "/disk/ml/datasets/KITTI/object/data/training/calib/"
    my_db_path = "/disk/users/sc468/no_backup/my_kitti/"
    # build(pcd_path, label_path, calib_path, my_db_path)
    # data_augmentation(my_db_path)
    # generate_train_test_split_file(my_db_path)

    class_hwl = get_mean_hwl(pcd_path, label_path, calib_path)
    print(class_hwl)