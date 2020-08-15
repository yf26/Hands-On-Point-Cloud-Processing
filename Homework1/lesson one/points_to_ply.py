# 本文件功能是把ModelNet中的.off文件转存成.ply文件
# 如果下载的不是.off文件，则补需要执行此文件

import os
import time

import numpy as np
from plyfile import PlyData
from plyfile import PlyElement


# 功能：从off文件中读取点云信息
# 输入：
#     filename:off文件名
def read_off(filename):
    points = []
    faces = []
    with open(filename, 'r') as f:
        first = f.readline()
        if (len(first) > 4):
            n, m, c = first[3:].split(' ')[:]
        else:
            n, m, c = f.readline().rstrip().split(' ')[:]
        n = int(n)
        m = int(m)
        for i in range(n):
            value = f.readline().rstrip().split(' ')
            points.append([float(x) for x in value])
        for i in range(m):
            value = f.readline().rstrip().split(' ')
            faces.append([int(x) for x in value])
    points = np.array(points)
    faces = np.array(faces)
    return points, faces


def read_off_only_pts(filename):
    points = []
    with open(filename, 'r') as f:
        first = f.readline()
        if (len(first) > 4):
            n, m, c = first[3:].split(' ')[:]
        else:
            n, m, c = f.readline().rstrip().split(' ')[:]
        n = int(n)

        for i in range(n):
            value = f.readline().rstrip().split(' ')
            points.append([float(x) for x in value])

    points = np.array(points)
    return points


def read_txt_pts_v1(filename):
    with open(filename, 'r') as f:
        pts = np.loadtxt(filename, dtype='float', delimiter=',', usecols=[0,1,2])
    return pts


# faster than v1
def read_txt_pts_v2(filename):
    pts = []
    with open(filename, 'r') as f:
        for i in range(10000):
            temp = f.readline().split(',')[0:3]
            pts.append([float(x) for x in temp])
    return np.array(pts)


def read_txt_pts_nmls(filename):
    points_normals = []
    with open(filename, 'r') as f:
        for i in range(10000):
            temp = f.readline().strip().split(',')
            points_normals.append([float(x) for x in temp])
    result = np.array(points_normals)
    return result


# 功能：把点云信息写入ply文件，只写points
# 输入：
#     pc:点云信息
#     filename:文件名
def export_ply(pc, filename):
    vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][1], pc[i][2])
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
    ply_filename = filename[:-4] + '.ply'
    ply_out.write(ply_filename)


def export_pts_nmls_ply(pc, filename):
    vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][1], pc[i][2], pc[i][3], pc[i][4], pc[i][5])
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
    ply_filename = filename[:-4] + '.ply'
    ply_out.write(ply_filename)


# 功能：把ModelNet数据集文件从.off格式改成.ply格式，只包含points
# 输入：
#     ply_data_dir: ply文件的存放路径
#     off_data_dir: off文件的存放地址
def write_ply_points_only_from_off(ply_data_dir, off_data_dir):
    cat = os.listdir(off_data_dir)
    for i in range(len(cat)):
        if not os.path.exists(os.path.join(ply_data_dir, cat[i], 'train')):
            os.makedirs(os.path.join(ply_data_dir, cat[i], 'train'))
        if not os.path.exists(os.path.join(ply_data_dir, cat[i], 'test')):
            os.makedirs(os.path.join(ply_data_dir, cat[i], 'test'))
    for i in range(len(cat)):
        print('writing ', i + 1, '/', len(cat), ':', cat[i])

        filenames = os.listdir(os.path.join(off_data_dir, cat[i], 'train'))
        for x in filenames:
            filename = os.path.join(off_data_dir, cat[i], 'train', x)
            out = os.path.join(ply_data_dir, cat[i], 'train', x)
            points = read_off_only_pts(filename)
            export_ply(points, out)

        filenames = os.listdir(os.path.join(off_data_dir, cat[i], 'test'))
        for x in filenames:
            filename = os.path.join(off_data_dir, cat[i], 'test', x)
            out = os.path.join(ply_data_dir, cat[i], 'test', x)
            points = read_off_only_pts(filename)
            export_ply(points, out)


def write_ply_points_only_from_txt(ply_data_dir, txt_data_dir):
    cat = os.listdir(txt_data_dir)
    for i in range(len(cat)):
        if not os.path.exists(os.path.join(ply_data_dir, cat[i])):
            os.makedirs(os.path.join(ply_data_dir, cat[i]))
    for i in range(len(cat)):
        print('writing ', i + 1, '/', len(cat), ':', cat[i])
        filenames = os.listdir(os.path.join(txt_data_dir, cat[i]))
        for x in filenames:
            # print("\tprocessing {}".format(x))
            filename = os.path.join(txt_data_dir, cat[i], x)
            out = os.path.join(ply_data_dir, cat[i], x)
            points = read_txt_pts_v2(filename)
            export_ply(points, out)


def write_ply_points_normals_from_txt(ply_data_dir, txt_data_dir):
    cat = os.listdir(txt_data_dir)
    for i in range(len(cat)):
        if not os.path.exists(os.path.join(ply_data_dir, cat[i])):
            os.makedirs(os.path.join(ply_data_dir, cat[i]))
    for i in range(len(cat)):
        print('writing ', i + 1, '/', len(cat), ':', cat[i])
        filenames = os.listdir(os.path.join(txt_data_dir, cat[i]))
        for x in filenames:
            # print("\tprocessing {}".format(x))
            filename = os.path.join(txt_data_dir, cat[i], x)
            out = os.path.join(ply_data_dir, cat[i], x)
            points_normals = read_txt_pts_nmls(filename)
            export_pts_nmls_ply(points_normals, out)


# 功能：把点云信息写入ply文件，包括points和faces
# 输入：
#     pc：points的信息
#     fc: faces的信息
#     filename:文件名
def export_ply_points_faces(pc, fc, filename):
    vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    face = np.zeros(fc.shape[0], dtype=[('vertex_indices', 'i4', (3,))])

    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][1], pc[i][2])

    for i in range(fc.shape[0]):
        face[i] = (fc[i, 1], fc[i, 2], fc[i, 3])

    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices']),
                       PlyElement.describe(face, 'face', comments=['faces'])])

    ply_filename = filename[:-4] + '.ply'
    ply_out.write(ply_filename)


# 功能：把ModelNet数据集文件从.off格式改成.ply格式，包含points和faces
# 输入：
#     ply_data_dir: ply文件的存放路径
#     off_data_dir: off文件的存放地址
def write_ply_points_faces_from_off(ply_data_dir, off_data_dir):
    cat = os.listdir(off_data_dir)
    for i in range(len(cat)):
        if not os.path.exists(os.path.join(ply_data_dir, cat[i], 'train')):
            os.makedirs(os.path.join(ply_data_dir, cat[i], 'train'))
        if not os.path.exists(os.path.join(ply_data_dir, cat[i], 'test')):
            os.makedirs(os.path.join(ply_data_dir, cat[i], 'test'))
    for i in range(len(cat)):
        print('writing ', i + 1, '/', len(cat), ':', cat[i])
        filenames = os.listdir(os.path.join(off_data_dir, cat[i], 'train'))
        for j, x in enumerate(filenames):
            filename = os.path.join(off_data_dir, cat[i], 'train', x)
            out = os.path.join(ply_data_dir, cat[i], 'train', x)
            points, faces = read_off(filename)
            export_ply_points_faces(points, faces, out)
        filenames = os.listdir(os.path.join(off_data_dir, cat[i], 'test'))
        for j, x in enumerate(filenames):
            filename = os.path.join(off_data_dir, cat[i], 'test', x)
            out = os.path.join(ply_data_dir, cat[i], 'test', x)
            points, faces = read_off(filename)
            export_ply_points_faces(points, faces, out)


def convert_off_to_ply():
    # ply目标文件产生路径
    ply_data_dir = '/home/yu/dataset/ModelNet40_ply_complete'
    # off文件所在路径
    off_data_dir = '/home/yu/dataset/ModelNet40'
    write_ply_points_only_from_off(ply_data_dir, off_data_dir)


def convert_txt_to_ply():
    ply_data_dir = '/home/yu/dataset/modelnet40_normal_resampled_ply'
    txt_data_dir = '/home/yu/dataset/modelnet40_normal_resampled'
    write_ply_points_normals_from_txt(ply_data_dir, txt_data_dir)


if __name__ == '__main__':
    # main()
    convert_txt_to_ply()
