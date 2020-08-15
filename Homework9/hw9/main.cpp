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

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double, std::milli> Duration;

using namespace std::chrono_literals;

std::vector<float> doRegistration(std::string& path, std::string idx_src, std::string idx_tar)
{
    PointCloud cloud_source;
    PointCloud cloud_target;
    NormalCloud normals_source;
    NormalCloud normals_target;
    // TODO check 6, 178 / 650, 177
    std::string path_source = path + idx_src + ".bin";
    std::string path_target = path + idx_tar + ".bin";

    auto read_start = Clock::now();
    float voxel_size = 0.3;
    readBinaryAndVoxelDown(path_source, cloud_source, normals_source, voxel_size);
    readBinaryAndVoxelDown(path_target, cloud_target, normals_target, voxel_size);
//    readBinaryAndNormalSpaceDown(path_source, cloud_source, normals_source);
//    readBinaryAndNormalSpaceDown(path_target, cloud_target, normals_target);
    auto read_end = Clock::now();
    Duration read_time = read_end - read_start;
    DEBUG("Cloud reading and preprocessing takes " << read_time.count() << " ms");

//    DEBUG("test cloud:");
//    Eigen::MatrixXf cloud_mat = cloud_source.getMatrixXfMap().transpose();
//    DEBUG(cloud_mat);
//    DEBUG(cloud_mat.rows() << " x " << cloud_mat.cols());
//
//    DEBUG("test normals:");
//    Eigen::MatrixXf normals_mat = normals_source.getMatrixXfMap().transpose();
//    DEBUG(normals_mat);
//    DEBUG(normals_mat.rows() << " x " << normals_mat.cols());

//    pcl::visualization::PCLVisualizer::Ptr viewer_test(new pcl::visualization::PCLVisualizer("3D Viewer"));
//    viewer_test->addPointCloudNormals<PointT, NormalT>(cloud_source.makeShared(), normals_source.makeShared(), 1, 0.5, "cloud_source");
//    viewer_test->addPointCloudNormals<PointT, NormalT>(cloud_target.makeShared(), normals_target.makeShared(), 1, 0.5, "cloud_target");
//    viewer_test->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0., 0., 1., "cloud_source");
//    viewer_test->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 1., "cloud_target");
//    viewer_test->setBackgroundColor(0, 0, 0);
//    while (!viewer_test->wasStopped())
//    {
//        viewer_test->spinOnce ();
//        std::this_thread::sleep_for(100ms);
//    }

//    Eigen::Vector3f translation = {-6.2167,-2.64747,-0.102533};
//    Eigen::Quaternionf rotation = {0.890111,-0.0054595,0.00239728,-0.455707};


    Registration reg;

//    float   iss_salient_radius = voxel_size * 3;
//    float   iss_non_max_radius = voxel_size * 3;
//    float   iss_gamma_21 = 0.52;
//    float   iss_gamma_32 = 0.52;
//    int     iss_min_neighbors = 6;
//    int     iss_thread = 8;
//    reg.setISSparams(iss_salient_radius, iss_non_max_radius, iss_gamma_21, iss_gamma_32, iss_min_neighbors, iss_thread);

    float   harris3d_radius = voxel_size * 2;
    float   harris3d_nms_threshold = 1e-8/*0.5e-5*/;
    int     harris3d_threads = 4;
    bool    harris3d_is_nms = true;
    bool    harris3d_is_refine = false;
    reg.setHarris3Dparams(harris3d_radius, harris3d_nms_threshold, harris3d_threads, harris3d_is_nms, harris3d_is_refine);

//    reg.setSHOTparams(0.4);

    reg.setFPFHparams(voxel_size * 4);

    reg.setRANSACparams(80000, voxel_size * 4, 10, 0.5);

    int     icp_normal_bins = 10;
    size_t  icp_sampled_size = 4000;
    float   icp_max_corres_dist = 1;
    size_t  icp_max_iter = 800;
//    float   icp_loss_epsilon = 1e-5; // for icp point to plane
    float   icp_loss_epsilon = 1e-8; // for icp point to point

    reg.setICPparams(icp_normal_bins, icp_sampled_size, icp_max_corres_dist, icp_max_iter, icp_loss_epsilon);

    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    reg.compute(cloud_source, cloud_target, normals_source, normals_target, R, t);
    DEBUG("\nFinal det(R) = " << R.determinant());

    Eigen::Matrix4f T = Eigen::MatrixXf::Identity(4, 4);
    T.block(0, 0, 3, 3) = R;
    T.block(0, 3, 3, 1) = t;
//    DEBUG(T);

    pcl::transformPointCloud(cloud_source, cloud_source, T);

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

    Eigen::Quaternionf q(R);
    return std::vector<float>{t(0), t(1), t(2), q.w(), q.x(), q.y(), q.z()};
}


void processDataSet(std::string file_name)
{
    std::string reg_path = "/home/yu/0_point_cloud_learn/DataSet/registration_dataset/reg_result.txt";
    std::string path = "/home/yu/0_point_cloud_learn/DataSet/registration_dataset/point_clouds/";

    std::vector<std::vector<std::string>> idx_tar_src;
    idx_tar_src.reserve(350);
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
        idx_tar_src.emplace_back(std::vector<std::string>{temp[0], temp[1]});
    }
    file.close();

    DEBUG("Read " << idx_tar_src.size() << " cloud pairs!");

    std::ofstream out_file("../" + file_name);
    out_file << "idx1,idx2,t_x,t_y,t_z,q_w,q_x,q_y,q_z\n"
             << "643,456,-4.3284934319736,-5.76281697543767,-0.74519599142,0.999709198802746,0.0168596885066202,0.000616225985072419,-0.0172304671488069\n"
             << "0,456,0.960934483080504,1.8221339090776,-0.174172942948,0.543717415764284,-0.00582557221576676,0.0111554518084263,-0.839173992922753\n"
             << "645,189,-0.666134774322387,4.30285330421818,-0.121591196312,0.765295644462435,0.00352529959341506,0.00844996765427172,0.643613818120963\n";
    for (size_t i = 4; i < idx_tar_src.size(); i++)
    {
        DEBUG("\n\nProcessing " << idx_tar_src[i][1] << " - " << idx_tar_src[i][0] << ", " << i << "/" << idx_tar_src.size());
        std::vector<float> result = doRegistration(path, idx_tar_src[i][1], idx_tar_src[i][0]);

        out_file << idx_tar_src[i][0] << ","
                 << idx_tar_src[i][1] << ","
                 << result[0] << ","
                 << result[1] << ","
                 << result[2] << ","
                 << result[3] << ","
                 << result[4] << ","
                 << result[5] << ","
                 << result[6] << "\n";
    }
    out_file.close();
    DEBUG("Done!");

//    std::random_device rd;
//    std::mt19937 mt(rd());
//    std::uniform_int_distribution<size_t> dist(0, idx_tar_src.size() - 1);
//    while(true)
//    {
//        size_t i = dist(mt);
//        DEBUG("\n\nProcessing " << idx_tar_src[i][1] << " - " << idx_tar_src[i][0]);
//        doRegistration(path, idx_tar_src[i][1], idx_tar_src[i][0]);
//    }
}


int main()
{
    processDataSet("my_reg_result_v3.txt");
    return 0;
}

