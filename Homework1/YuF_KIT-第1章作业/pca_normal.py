# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import time
import numpy as np
from pyntcloud import PyntCloud
import mylib

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    center_point = np.sum(data, axis=0) / data.shape[0]
    centered_data = np.subtract(data, center_point)
    XTX = centered_data.transpose().dot(centered_data)
    # XTX = np.cov(data.transpose())
    eigenvalues, eigenvectors = np.linalg.eig(XTX)
    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def PCA_faster(data):
    center_point = np.sum(data, axis=0) / data.shape[0]
    centered_data = np.subtract(data, center_point)
    XTX = centered_data.transpose().dot(centered_data)
    # using FastEigen3x3 with assumption that input matirix is 3x3 and symmetric
    normal = mylib.FastEigen3x3(XTX)
    return normal

def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file("/home/yu/3D Point Cloud Processing/DataSet/MyDataSet/airplane/test/airplane_0645.ply")
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 0] #点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    # visualize the 3 principle axis on the point cloud
    # and length means the relative deviation of point cloud along each axis
    center_point = point_cloud_pynt.centroid
    frame_points = np.append(
        [center_point],
        center_point + np.diag(500 *w / np.linalg.norm(w)).dot(v.transpose()), axis=0
    )
    frame_lines = [
        [0, 1],
        [0, 2],
        [0, 3],
    ]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    pca_frame = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(frame_points),
        lines=o3d.utility.Vector2iVector(frame_lines),
    )
    pca_frame.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud_o3d, pca_frame])

    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    # 作业2
    # 屏蔽开始
    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
    normals = np.zeros((points.shape[0], 3))
    start = time.time()
    points = np.asarray(points)
    for i in range(points.shape[0]):
        num, idx_list, _ = pcd_tree.search_hybrid_vector_3d(points[i, :], radius=5, max_nn=10)
        if num >= 3:
            # using cpp lib
            # normal = PCA_faster(points[idx_list, :])
            # normals[i, :] = normal
            # using self implementation
            _, eig_vectors = PCA(points[idx_list, :])
            normals[i, :] = np.real(eig_vectors[:, 2])
    print("Normals estimation takes {}s".format(time.time() - start))
    # 屏蔽结束
    # normals = np.array(normals, dtype=np.float64)
    # # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])

    # # using o3d.geometry.PointCloud.estimate_normals
    # start = time.time()
    # point_cloud_o3d.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #         radius=5, max_nn=10
    #     )
    # )
    # print("Normals estimation takes {}s".format(time.time() - start))
    # o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
