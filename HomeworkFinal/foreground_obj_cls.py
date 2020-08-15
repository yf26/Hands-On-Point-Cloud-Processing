import importlib
import sys
import os
import time
import numpy as np
import open3d as o3d
import torch
from dataset_building.ground_detection_SVD import raw_kitti_bin_reader, \
    pcd_preprocessing, ground_detection_on3segs, plot_clusters, visualize_pcd
from data_utils.DataLoader import farthest_point_sample

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
print(ROOT_DIR)


class Object:
    def __init__(self, indices, cluster_idx, cls):
        self.indices = indices
        self.cluster_idx = cluster_idx
        self.cls = cls


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    # m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # pc = pc / m
    return pc


# def get_bbox_params_car(points):
#
#     centroid = np.mean(points, axis=0)
#     centerd_xy = points[:, :2] - centroid[:, :2]
#     XTX = centerd_xy.transpose().dot(centerd_xy)
#     eigval, eigvec = np.linalg.eig(XTX)
#     # TODO
#     # sort = eigval.argsort()[::-1]
#     # eigvec = eigvec[:, sort]
#
#     hwl = np.zeros(3)
#     for i in range(3):
#         prin_axis = eigvec[]
#     proj = centered_points.dot(prin_axis)
#     return np.max(proj) - np.min(proj)


def get_expand_z(points):
    return np.max(points[:, 2]) - np.min(points[:, 2])


def get_min_z_value(points):
    return np.min(points[:, 2])


def get_max_z_value(points):
    return np.max(points[:, 2])


def get_mean_z_value(points):
    return np.mean(points[:, 2])


def plot_classified_clusters(g_pcd, fg_pcd, cluster_per_point, predicts, bbox_list):
    colors = {-1: [0.6, 0.6, 0.6],
              0: [1, 0, 1],
              1: [1, 0, 0],
              2: [0, 0, 1],
              3: [0.6, 0.6, 0.6]}

    assert isinstance(g_pcd, o3d.geometry.PointCloud)
    assert isinstance(fg_pcd, o3d.geometry.PointCloud)

    fg_colors = []
    for c in cluster_per_point:
        if c == -1:
            fg_colors.append(colors[c])
            continue
        cls = predicts[c]
        fg_colors.append(colors[cls])
    fg_pcd.colors = o3d.utility.Vector3dVector(fg_colors)

    geometries = bbox_list
    geometries.append(g_pcd)
    geometries.append(fg_pcd)
    o3d.visualization.draw_geometries(geometries)
    # o3d.visualization.draw_geometries([g_pcd, fg_pcd])


if __name__ == "__main__":
    print("\nReading and Preprocessing the raw point cloud...")
    start = time.time()
    # path = "/media/yu/学习/DataSet/KITTI/object/testing/velodyne/"
    path = "/disk/ml/datasets/KITTI/object/data/testing/velodyne/"
    # 000006
    points = raw_kitti_bin_reader(path, "000388")[:, :3]
    points = pcd_preprocessing(points)
    # visualize_pcd(points)
    print("Preprocessing takes {}ms".format(1000 * (time.time() - start)))

    print("\nGround detection...")
    start = time.time()
    ground_idx, foreground_idx = ground_detection_on3segs(points)
    ground_z = get_mean_z_value(points[ground_idx, :])
    print("Ground z = {}".format(ground_z))
    # _, ground_idx = ground_detection(points, np.array(range(points.shape[0])), 10, 10000, 0.15)
    print("Ground detection takes {}ms".format(1000 * (time.time() - start)))

    # transform np.array to open3d point cloud for clustering and visualization
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(points[ground_idx])
    ground_pcd.paint_uniform_color([0.85, 0.85, 0.85])

    foreground_pcd = o3d.geometry.PointCloud()
    foreground_pcd.points = o3d.utility.Vector3dVector(points[foreground_idx])

    print("\nClustering foreground...")
    start = time.time()
    cluster_per_point = foreground_pcd.cluster_dbscan(0.5, 8, print_progress=True)
    print("Clustering takes {}ms".format(1000 * (time.time() - start)))
    plot_clusters(ground_pcd, foreground_pcd, np.asarray(cluster_per_point))

    print("\nDownsampling and classifing the clustered objects...")
    # load pretrained model
    num_class = 4
    model_name = "pointnet2_cls_ssg"

    # model_path = str(ROOT_DIR) + \
    #              "/report/augmentated_nll_loss_64pts/2020-06-26_18-04/checkpoints/best_model.pth"

    model_path = str(ROOT_DIR) + \
                 "/report/augmentated_nll_loss_256pts_original_size/2020-06-28_10-47/checkpoints/best_model.pth"

    # model_path = str(ROOT_DIR) + \
    #              "/report/augmentated_nll_loss_256pts/2020-06-27_22-30/checkpoints/best_model.pth"
    MODEL = importlib.import_module(model_name)
    checkpoint = torch.load(model_path)
    classifier = MODEL.get_model(num_class, normal_channel=False).cuda()
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    cluster_indices_dict = {}
    for idx, cluster in enumerate(cluster_per_point):
        try:
            cluster_indices_dict[cluster].append(idx)
        except KeyError:
            cluster_indices_dict[cluster] = [idx]

    foreground_points = points[foreground_idx]
    object_list = []
    pred_final = np.zeros(len(cluster_indices_dict) - 1)
    for cluster in cluster_indices_dict:
        if cluster == -1:
            continue
        obj_indices = cluster_indices_dict[cluster]
        obj_size = len(obj_indices)
        obj_pts = foreground_points[obj_indices, :]

        min_z = get_min_z_value(obj_pts)
        obj_z_threshold = 0.5
        if min_z - ground_z > obj_z_threshold:
            pred_final[cluster] = 3
            continue

        z_expand = get_expand_z(obj_pts)
        if z_expand < 1 or z_expand > 2.3:
            pred_final[cluster] = 3
            continue

        npoints = 256
        if obj_size > npoints:
            # select npoints using FPS
            obj_pts = farthest_point_sample(obj_pts, npoints)
        else:
            # select npoints randomly
            append_choice = np.random.choice(obj_size, npoints - obj_size, replace=True)
            point_append = obj_pts[append_choice, :]
            obj_pts = np.append(obj_pts, point_append, axis=0)
        obj_pts = pc_normalize(obj_pts)

        input = torch.zeros([1, npoints, 3], requires_grad=False)
        input[0, ...] = torch.from_numpy(obj_pts)
        with torch.no_grad():
            pred, _ = classifier(input.transpose(2, 1).cuda())
            pred_choice = pred.data.max(1)[1].cpu()
            # print(pred_choice)
        pred_final[cluster] = pred_choice.item()

    bbox_list = []
    for cluster, cls in enumerate(pred_final):
        if cls in [0, 1, 2]:
            obj_indices = cluster_indices_dict[cluster]
            obj_pts = foreground_points[obj_indices, :]
            bbox_o3d = o3d.geometry.AxisAlignedBoundingBox()
            bbox_o3d = bbox_o3d.create_from_points(o3d.utility.Vector3dVector(obj_pts))
            # print(bbox_o3d.get_center())
            # print(bbox_o3d.get_max_bound())
            bbox_list.append(bbox_o3d)

    plot_classified_clusters(ground_pcd, foreground_pcd, np.asarray(cluster_per_point), pred_final, bbox_list)
