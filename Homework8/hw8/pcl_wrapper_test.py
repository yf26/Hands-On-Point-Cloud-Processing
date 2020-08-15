import PCLKeypoint as pcl
import open3d as o3d
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt


def pick_points(pcd):
    print("")
    print("********** Key Point Selection Guide **********")
    print("0) Press = to make keypoints bigger for better selection")
    print("1) Press [shift + left click] to pick the keypoints that you want to check the descriptor histogram")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


if __name__ == "__main__":
    point_cloud_pynt = PyntCloud.from_file("table_0020.ply")
    input_cloud = point_cloud_pynt.to_instance("open3d", mesh=False)
    input_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    assert isinstance(input_cloud, o3d.geometry.PointCloud)
    points = np.asarray(input_cloud.points)
    normals = np.asarray(input_cloud.normals)
    print("Calculating ISS keypoints...")
    keypoints = pcl.keypointIss(points, 0.16, 0.16, 0.9, 0.9, 5, 4)
    print("Keypoints number = {}".format(keypoints.shape[0]))

    print("Calculating descriptors...")
    FPFH_descriptors = pcl.featureFPFH33(points, keypoints, 10, 0.16)
    FPFH_Normal_descriptors = pcl.featureFPFH33WithNormal(points, normals, keypoints, 0.16)
    SHOT_descriptors = pcl.featureSHOT352(points, keypoints, 10, 0.25)
    SHOT_Normal_descriptors = pcl.featureSHOT352WithNormal(points, normals, keypoints, 0.25)

    # visualize input cloud with keypoints
    key_cloud = o3d.geometry.PointCloud()
    key_cloud.points = o3d.utility.Vector3dVector(keypoints)
    key_cloud.paint_uniform_color([1, 0, 1])
    o3d.visualization.draw_geometries([input_cloud, key_cloud])

    # visualize only keypoints for better selection
    picked_idx = pick_points(key_cloud)
    picked_idx = list(dict.fromkeys(picked_idx))

    # visualize the histograms of selected keypoints
    descriptors_map = {0: FPFH_descriptors, 1: FPFH_Normal_descriptors, 2: SHOT_descriptors, 3: SHOT_Normal_descriptors}
    titles = {0: "FPFH", 1: "FPFH with Normals", 2: "SHOT", 3: "SHOT with Normals"}
    colors = {0: "r", 1: "g", 2: "b", 3: "m"}

    height = len(picked_idx) * 2.5
    plt.figure(figsize=[10, height])
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)
    for j in range(4):
        for i, idx in enumerate(picked_idx):
            plt.subplot(len(picked_idx), 4, j+1+4*i)
            if i == 0:
                plt.title(titles[j], size=15)
            plt.plot(descriptors_map[j][idx, :], colors[j])
            plt.xticks(())
            plt.yticks(())
            plt.xlabel("Point Num: {}".format(i + 1))
    plt.show()
