#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <boost/algorithm/string.hpp>
#include <Eigen/Geometry>
#include "registration.hpp"
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#define DEBUG(x) std::cout << x << std::endl

using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;
using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;

using NormalT = pcl::Normal;
using NormalCloud = pcl::PointCloud<pcl::Normal>;
using NormalCloudPtr = pcl::PointCloud<pcl::Normal>::Ptr;

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double, std::milli> Duration;

using namespace std::chrono_literals;

void testResults(std::string file_name)
{
    std::string reg_path = "../" + file_name;
    std::string path = "/home/yu/0_point_cloud_learn/DataSet/registration_dataset/point_clouds/";

    std::ifstream file(reg_path);
    if (not file.good())
    {
        std::cerr << "Read file " << reg_path << " failed!";
        exit(EXIT_FAILURE);
    }

    std::string line;
    while(getline(file, line))
    {
        std::vector<std::string> temp;
        boost::algorithm::split(temp, line, boost::is_any_of(","));

        if (temp[0] == "idx1")
            continue;

        std::string path_source = path + temp[1] + ".bin";
        std::string path_target = path + temp[0] + ".bin";

        PointCloud cloud_source;
        PointCloud cloud_target;
        NormalCloud normals_source;
        NormalCloud normals_target;
        float voxel_size = 0.3;
        readBinaryAndVoxelDown(path_source, cloud_source, normals_source, voxel_size);
        readBinaryAndVoxelDown(path_target, cloud_target, normals_target, voxel_size);

        Eigen::Vector3f t = {std::stof(temp[2]), std::stof(temp[3]), std::stof(temp[4])};
        Eigen::Quaternionf q = {std::stof(temp[5]), std::stof(temp[6]), std::stof(temp[7]), std::stof(temp[8])};

        pcl::transformPointCloud(cloud_source, cloud_source, t, q);

        pcl::visualization::PCLVisualizer viewer;
        viewer.addPointCloud<PointT>(cloud_source.makeShared(), "source");
        viewer.addPointCloud<PointT>(cloud_target.makeShared(), "target");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0., 0., 1., "source", 1);
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 1., "target", 1);
        viewer.setBackgroundColor(0, 0, 0);
        while (!viewer.wasStopped())
        {
            viewer.spinOnce(100);
            std::this_thread::sleep_for(100ms);
        }

    }
    file.close();
}


int main(int argc, char** argv)
{
    testResults(argv[1]);
    return 0;
}

