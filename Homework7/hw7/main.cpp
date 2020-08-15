#include <iostream>
#include <chrono>
#include <pcl/io/ply_io.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>

#include "iss_detector.hpp"

typedef pcl::PointXYZ PointT;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<float, std::milli> Duration;


void pclISSDetector(std::string& path)
{
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::io::loadPLYFile(path, *cloud);


    double test = 0.02;

    auto start = Clock::now();

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    pcl::ISSKeypoint3D<PointT, PointT> iss_detector;
    iss_detector.setSearchMethod(tree);
    iss_detector.setSalientRadius(6 * test);
    iss_detector.setNonMaxRadius(4 * test);
    iss_detector.setThreshold21(0.9);
    iss_detector.setThreshold32(0.9);
    iss_detector.setMinNeighbors(5);
    iss_detector.setNumberOfThreads(4);
    iss_detector.setInputCloud(cloud);

    pcl::PointCloud<PointT>::Ptr keys(new pcl::PointCloud<PointT>);
    iss_detector.compute(*keys);
    auto end = Clock::now();
    Duration duration = end - start;
    std::cout << "detection takes " << duration.count() << " ms" << std::endl;
    std::cout << "key points size : " << keys->size() << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud(cloud, "input");
    viewer->addPointCloud(keys, "keys");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0., 1., 0., "input");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7.5, "keys");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 0., "keys");

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    }
}

void myISSDetector(std::string& path)
{
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::io::loadPLYFile(path, *cloud);
    std::cout << "input cloud w x h = " << cloud->width << " x " << cloud->height << std::endl;

    size_t num_points = cloud->size();
    MyPCDType point_cloud(num_points, std::vector<float>(3));

    // TODO
    for (int i = 0; i < num_points; i++)
    {
        point_cloud[i][0] = cloud->points[i].x;
        point_cloud[i][1] = cloud->points[i].y;
        point_cloud[i][2] = cloud->points[i].z;
    }

    auto start = Clock::now();

    float test = 0.02;
    ISSKeypoint iss_detector;
    iss_detector.useWeightedCovMat(true);
    iss_detector.setInputPointCloud(point_cloud);
    iss_detector.setLocalRadius(6 * test);
    iss_detector.setNonMaxRadius(4 * test);
    iss_detector.setThreshold(0.9, 0.9);
    iss_detector.setMinNeighbors(5);
    MyPCDType keypoints;
    iss_detector.compute(keypoints);

    auto end = Clock::now();
    Duration duration = end - start;
    std::cout << "detection takes " << duration.count() << " ms" << std::endl;
    std::cout << "key points size : " << keypoints.size() << std::endl;

    // TODO
    pcl::PointCloud<PointT>::Ptr keys(new pcl::PointCloud<PointT>);
    for (const auto& keypoint : keypoints)
    {
        keys->points.emplace_back(
            PointT(keypoint[0], keypoint[1], keypoint[2])
        );
    }

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud(cloud, "input");
    viewer->addPointCloud(keys, "keys");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0., 1., 0., "input");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7.5, "keys");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 0., "keys");

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    }


    // compute normal for keypoints
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal> ());
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
    norm_est.setKSearch(10);
    norm_est.setSearchSurface(cloud);
    norm_est.setInputCloud(cloud);
    norm_est.compute(*normals);

    // Create the FPFH estimation class, and pass the input dataset+normals to it
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(keys);
    fpfh.setInputNormals(normals);
    fpfh.setSearchSurface(cloud);
    // alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

    // Create an empty kdtree representation, and pass it to the FPFH estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr p_tree (new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh.setSearchMethod(p_tree);

    // Output datasets
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_descriptors (new pcl::PointCloud<pcl::FPFHSignature33> ());

    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    fpfh.setRadiusSearch(0.18);

    // Compute the features
    fpfh.compute(*fpfh_descriptors);

    std::cout << "fpfh_descriptors size " << fpfh_descriptors->size() << std::endl;
    
	for ( size_t i = 0; i < 33 ; i++ )
	{
		std::cout << fpfh_descriptors->points[0].histogram[i] << std::endl;
	}
	

}


int main()
{
    std::string path = "../test_data/airplane_0001.ply";
    myISSDetector(path);
	return 0;
}
