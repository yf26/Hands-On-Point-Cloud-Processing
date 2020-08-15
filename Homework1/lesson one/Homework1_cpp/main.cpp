#include <iostream>
#include <thread>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>

#define LOG(x) std::cout << x << std::endl

using namespace std::chrono_literals;

struct pcaResult
{
    Eigen::Matrix3f eigenVectors;
    Eigen::Vector3f eigenValues;
};

pcaResult PCA(const Eigen::MatrixXf& points)
{
    pcaResult result;
    Eigen::Vector3f center = points.colwise().mean();
    Eigen::MatrixXf centeredPoints = points.rowwise() - center.transpose();
    Eigen::Matrix3f XTX = centeredPoints.transpose() * centeredPoints;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(XTX);
    result.eigenVectors = eigenSolver.eigenvectors().real();
    result.eigenValues = eigenSolver.eigenvalues().real();
    return result; // eigenvalues from lowest to highest
}



int main()
{
    // read point cloud from .ply file
    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
    PointCloud::Ptr pointCloud(new PointCloud);

    pcl::io::loadPLYFile("/home/yu/3D Point Cloud Processing/DataSet/MyDataSet/airplane/test/airplane_0645.ply", *pointCloud);
    pointCloud->is_dense = false;
    std::size_t cloudSize = pointCloud->size();
    LOG("Cloud size = " << cloudSize);

    // transform points to eigen format
    Eigen::MatrixXf temp = pointCloud->getMatrixXfMap();
    Eigen::MatrixXf points = temp.block(0, 0, 3, temp.cols()).transpose();

    // PCA
    LOG("PCA on total point cloud...");
    pcaResult result = PCA(points);
    Eigen::Matrix3f eigenvectors = result.eigenVectors;
    Eigen::Vector3f eigenvalues = result.eigenValues;
    LOG("Eigen vectors = \n" << eigenvectors);
    LOG("Eigen values = \n" << eigenvalues);

    // preprocessing for visualization of principle axis
    Eigen::Vector3f stdv(sqrt(sqrt(eigenvalues(0))), sqrt(sqrt(eigenvalues(1))), sqrt(sqrt(eigenvalues(2))));
    Eigen::Vector3f center_temp = points.colwise().mean();
    pcl::PointXYZ center(center_temp(0), center_temp(1), center_temp(2));
    pcl::PointXYZ p1(center_temp(0)+stdv(2)*eigenvectors(0, 2), center_temp(1)+stdv(2)*eigenvectors(1, 2), center_temp(2)+stdv(2)*eigenvectors(2, 2));
    pcl::PointXYZ p2(center_temp(0)+stdv(1)*eigenvectors(0, 1), center_temp(1)+stdv(1)*eigenvectors(1, 1), center_temp(2)+stdv(1)*eigenvectors(2, 1));
    pcl::PointXYZ p3(center_temp(0)+stdv(0)*eigenvectors(0, 0), center_temp(1)+stdv(0)*eigenvectors(1, 0), center_temp(2)+stdv(0)*eigenvectors(2, 0));

    // visualization of PCA resualts
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0.1, 0.1, 0.1);
    viewer->addPointCloud(pointCloud);
    viewer->addLine(center, p1, 1, 0, 0, "pca1");
    viewer->addLine(center, p2, 0, 1, 0, "pca2");
    viewer->addLine(center, p3, 0, 0, 1, "pca3");
    while (!viewer->wasStopped())
    {
        viewer->spinOnce (100);
        std::this_thread::sleep_for(100ms);
    };

    // Normal estimation based on PCA
    // K nearest neighbor search
    LOG("Normal estimation...");
    int K = 10;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(pointCloud);
    Eigen::Matrix<float, -1, -1> neighbors;
    neighbors.resize(K, 3);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    std::vector<int> searchIdx(K);
    std::vector<float> searchSqrDistances(K);
    pcaResult temp_result;

    // normal estimation for each point
    auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < cloudSize; i++)
    {
        kdtree.nearestKSearch(i, K, searchIdx, searchSqrDistances);
        for (unsigned short j = 0; j < K; j++)
        {
            neighbors.row(j) = points.row(searchIdx[j]);
        }

        temp_result = PCA(neighbors);

        cloud_normals->push_back(pcl::Normal(temp_result.eigenVectors.col(0)(0), temp_result.eigenVectors.col(0)(1), temp_result.eigenVectors.col(0)(2)));
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    LOG("Normal estimation takes " << duration.count() << " ms.");

    // visualization of the normal estimation result
    pcl::visualization::PCLVisualizer::Ptr viewer1 (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer1->setBackgroundColor (0.1, 0.1, 0.1);
    viewer1->addPointCloud(pointCloud, "cloud");
    viewer1->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(pointCloud, cloud_normals, 1, 5, "normals", 0);
    viewer1->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "normals");
    viewer1->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "normals");
    while (!viewer1->wasStopped())
    {
        viewer1->spinOnce (100);
        std::this_thread::sleep_for(100ms);
    }

    return 0;
}