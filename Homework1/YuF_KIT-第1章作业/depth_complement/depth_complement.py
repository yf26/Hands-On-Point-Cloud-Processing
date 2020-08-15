import open3d as o3d
import os
import numpy as np
import math
import time

def find_correspondent_file(index, path_root):
    '''
    This function finds the corresponding image, depth image, intrinsics and ground truth from the dataset
    :param index: an integer to specify which image in the folder ".../image/" you want
    :param path_root: the root which contains the 4 folders, "groundtruth_depth", "velodyne_raw" and so on
    :return: the path of the 4 files
    '''
    image_name_list = os.listdir(os.path.join(path_root, 'image'))
    file_name = image_name_list[index]
    result = [os.path.join(path_root, 'image', file_name)]
    for folders in os.listdir(path_root):
        if folders != "image" and folders != 'intrinsics':
            temp = file_name
            new_name = temp.replace("image", str(folders), 1)
            result.append(os.path.join(path_root, folders, new_name))
        elif folders == 'intrinsics':
            temp = file_name
            new_name = temp.replace("png", "txt", 1)
            result.append(os.path.join(path_root, folders, new_name))
    return result


def gaussian(x, sigma):
    # return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))
    return math.exp(- (x ** 2) / (2 * sigma ** 2))


def bilateral_filter(intensity_crop, depth_crop, sigma):
    '''
    apply bilateral filter on the intensity image crop and depth crop according to the field size
    :param intensity_crop:
    :param depth_crop:
    :param sigma:
    :return:
    '''
    field_size = intensity_crop.shape[0]
    weight_sum = 0
    result = 0
    center_u = int(field_size / 2)
    center_v = center_u
    for u in range(field_size):
        for v in range(field_size):
            distance = math.sqrt((center_u - u) ** 2 + (center_v - v) ** 2)
            intensity_diff = intensity_crop[center_v, center_u] - intensity_crop[v, u]
            weight = gaussian(distance, sigma) * gaussian(intensity_diff, sigma)
            weight_sum = weight_sum + weight
            result = result + weight * depth_crop[v, u]
    if weight_sum == 0:
        return 0
    return result / weight_sum


def apply_bilateral_filter(intensity, depth, field_size, sigma):
    result = depth
    width = intensity.shape[1]
    height = intensity.shape[0]
    half_field = int(field_size / 2)

    for u in range(half_field, width - half_field):
        for v in range(half_field, height - half_field):
            # print(u, v)
            intensity_crop = intensity[v - half_field: v + half_field + 1, u - half_field: u + half_field + 1]
            # print(intensity_crop.shape)
            depth_crop = depth[v - half_field: v + half_field + 1, u - half_field: u + half_field + 1]
            result[v, u] = bilateral_filter(intensity_crop, depth_crop, sigma)
    return o3d.geometry.Image(np.array(result))

def get_zero_elem_number(depth):
    depth = np.asarray(depth)
    (unique_elem, counts) = np.unique(depth, return_counts=True)
    return counts[0]

if __name__ == '__main__':
    path_root = "/home/yu/3D Point Cloud Processing/DataSet/kitti/depth complement/"
    image_index = 20 # max 1000
    path_image, path_gt, path_intrinsic, path_depth = find_correspondent_file(image_index, path_root)
    print("The chosen image is:")
    print(path_image)
    image_raw = o3d.io.read_image(path_image)
    depth_raw = o3d.io.read_image(path_depth)
    depth_gt = o3d.io.read_image(path_gt)

    with open(path_intrinsic, "r") as f:
        params = [float(x) for x in f.readline().split()]

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=int(image_raw.get_max_bound()[0]),
        height=int(image_raw.get_max_bound()[1]),
        fx=params[0], fy=params[4], cx=params[2], cy=params[5]
    )

    # rotate the camera with extrinsic matrix, so that the point cloud isn't reversal
    extrinsic = np.array(
        [[1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]])

    # o3d.visualization.draw_geometries(
    #     [o3d.geometry.PointCloud.create_from_depth_image(depth_raw, camera_intrinsics, extrinsic)])
    # o3d.visualization.draw_geometries(
    #     [o3d.geometry.PointCloud.create_from_depth_image(depth_gt, camera_intrinsics, extrinsic)])

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_raw, depth_raw, convert_rgb_to_intensity=False, depth_scale=1000, depth_trunc=50
    )
    # ground truth rgbd image
    rgbd_image_gt = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_raw, depth_gt, convert_rgb_to_intensity=False, depth_scale=1000, depth_trunc=50
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics, extrinsic)
    # ground truth point cloud
    pcd_gt = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_gt, camera_intrinsics, extrinsic)

    print("Visulization of raw point cloud...")
    o3d.visualization.draw_geometries([pcd])


    # upsampling

    intensity_raw = np.mean(np.asarray(image_raw), axis=2)
    print("Upsampling start...")
    start = time.time()
    depth_filtered = apply_bilateral_filter(intensity_raw, np.asarray(depth_raw), 3, 0.42)
    print("upsampling takes {}s.".format(time.time() - start))

    rgbd_image_filtered = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_raw, depth_filtered, convert_rgb_to_intensity=False, depth_scale=1000, depth_trunc=50
    )
    pcd_filtered = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_filtered, camera_intrinsics, extrinsic)

    print("Visulization of upsampled point cloud...")
    o3d.visualization.draw_geometries([pcd_filtered])

    print("Visulization of ground truth point cloud...")
    o3d.visualization.draw_geometries([pcd_gt])

    print("Point number in raw point cloud: {}".format(np.asarray(pcd.points).shape[0]))
    print("Point number in filtered point cloud: {}".format(np.asarray(pcd_filtered.points).shape[0]))
    print("Point number in ground truth point cloud: {}".format(np.asarray(pcd_gt.points).shape[0]))
