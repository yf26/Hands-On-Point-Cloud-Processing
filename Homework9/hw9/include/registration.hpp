//
// Created by yu on 11.06.20.
//

#ifndef HW9_REGISTRATION_HPP
#define HW9_REGISTRATION_HPP

#include <fstream>
#include <random>
#include <limits>
#include <chrono>
#include <algorithm>
#include <eigen3/Eigen/Eigen>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/normal_space.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_omp.h>


#include <nanoflann.hpp>

#define DEBUG(x) std::cout << x << std::endl

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double, std::milli> Duration;

using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;
using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;

using NormalT = pcl::Normal;
using NormalCloud = pcl::PointCloud<pcl::Normal>;
using NormalCloudPtr = pcl::PointCloud<pcl::Normal>::Ptr;

using PointCloudwithNormal = pcl::PointCloud<pcl::PointNormal>;


void readBinaryAndVoxelDown(const std::string& fileName, PointCloud& cloud, NormalCloud& normals, float voxel_size);

void readBinaryAndNormalSpaceDown(const std::string& fileName, PointCloud& cloud, NormalCloud& normals);

void transformCloudInplace(PointCloud& cloud,
                           const Eigen::Matrix3f& R,
                           const Eigen::Vector3f& t);

void transformNormalsInplace(NormalCloud& cloud,
                             const Eigen::Matrix3f& R,
                             const Eigen::Vector3f& t);


class Registration
{
public:
    Registration() {}

    ~Registration()
    = default;

    void setISSparams(const float iss_salient_radius,
                      const float iss_non_max_radius,
                      const float iss_gamma_21,
                      const float iss_gamma_32,
                      const int iss_min_neighbors,
                      const int iss_threads)
    {
        m_iss_salient_radius = iss_salient_radius;
        m_iss_non_max_radius = iss_non_max_radius;
        m_iss_gamma_21 = iss_gamma_21;
        m_iss_gamma_32 = iss_gamma_32;
        m_iss_min_neighbors = iss_min_neighbors;
        m_iss_threads = iss_threads;
    }

    void setHarris3Dparams(const float harris3d_radius,
                           const float harris3d_nms_threshold,
                           const int harris3d_threads,
                           const bool harris3d_is_nms,
                           const bool harris3d_is_refine)
    {
        m_harris3d_radius =  harris3d_radius;
        m_harris3d_nms_threshold =  harris3d_nms_threshold;
        m_harris3d_threads =  harris3d_threads;
        m_harris3d_is_nms =  harris3d_is_nms;
        m_harris3d_is_refine =  harris3d_is_refine;
    }


    void setFPFHparams(const float fpfh_feature_radius)
    {
        m_fpfh_feature_radius = fpfh_feature_radius;
    }


    void setSHOTparams(const float shot_feature_radius)
    {
        m_shot_feature_radius = shot_feature_radius;
    }

    void setRANSACparams(const uint32_t max_iter,
                         const float dist_threshold,
                         const float angle_threshold,
                         const float corres_rejection_rate)
    {
        m_RANSAC_max_iter = max_iter;
        m_RANSAC_dist_threshold = dist_threshold;
        m_RANSAC_angle_threshold = angle_threshold;
        m_RANSAC_corres_rejection_rate = corres_rejection_rate;
    }

    void setICPparams(const int normal_bins,
                      const size_t sampled_size,
                      const float max_corres_dist,
                      const size_t max_iter,
                      const float loss_epsilon)
    {
        m_ICP_normal_bins = normal_bins;
        m_ICP_sampled_size = sampled_size;
        m_ICP_max_corres_dist = max_corres_dist;
        m_ICP_max_iter = max_iter;
        m_ICP_loss_epsilon = loss_epsilon;
    }

    void compute(const PointCloud& cloud_source,
                 const PointCloud& cloud_target,
                 const NormalCloud& normals_source,
                 const NormalCloud& normals_target,
                 Eigen::Matrix3f& R,
                 Eigen::Vector3f& t);

private:
    void getISSKeypoints(const PointCloud& input_cloud,
                         PointCloud& keypoints_cloud,
                         pcl::PointIndicesConstPtr& keypoints_indices);

    void getHarris3DKeypoints(const PointCloud& input_cloud,
                              const NormalCloud& input_normals,
                              PointCloud& keypoints_cloud);

    void getFPFH33Descriptors(const PointCloud& input_cloud,
                              const PointCloud& input_keypoints_cloud,
                              const NormalCloud& input_normals,
                              pcl::PointCloud<pcl::FPFHSignature33>& fpfh_descriptors);

    void getSHOT352Descirptors(const PointCloud& input_cloud,
                               const PointCloud& input_keypoints_cloud,
                               const NormalCloud& input_normals,
                               pcl::PointCloud<pcl::SHOT352>& shot_descriptors);

    void getKeypointsNormals(const NormalCloud& input_normals,
                             const pcl::PointIndicesConstPtr& keypoints_indices,
                             NormalCloud& extracted_normals);

    void normalSpaceSampling(const PointCloud& input_cloud,
                             const NormalCloud& input_normals,
                             PointCloud& sampled_cloud,
                             NormalCloud& sampled_normals);

    void VoxelGridSampling(const PointCloud& input_cloud,
                           const NormalCloud& input_normals,
                           PointCloud& sampled_cloud,
                           NormalCloud& sampled_normals);

    void RANSAC(const std::vector<std::vector<size_t>>& correspondences,
//                const NormalCloud& extracted_normals_source,
                const PointCloud& kp_cloud_source,
//                const NormalCloud& extracted_normals_target,
                const PointCloud& kp_cloud_target,
                Eigen::Matrix3f& R,
                Eigen::Vector3f& t);

    void findRANSACCorrespondencesInter(const pcl::PointCloud<pcl::FPFHSignature33>& fpfh_source,
                                  const pcl::PointCloud<pcl::FPFHSignature33>& fpfh_target,
                                  std::vector<std::vector<size_t>>& correspondences);

    void findRANSACCorrespondencesUnion(const pcl::PointCloud<pcl::FPFHSignature33>& fpfh_source,
                                  const pcl::PointCloud<pcl::FPFHSignature33>& fpfh_target,
                                  std::vector<std::vector<size_t>>& correspondences);

    void ICPpoint2plane(const Eigen::Matrix3f& init_R,
                        const Eigen::Vector3f& init_t,
                        const PointCloud& input_cloud_src,
                        const PointCloud& input_cloud_tar,
                        const NormalCloud& input_normals_src,
                        const NormalCloud& input_normals_tar,
                        Eigen::Matrix3f& R,
                        Eigen::Vector3f& t);

    void ICPpoint2point(const Eigen::Matrix3f& init_R,
                        const Eigen::Vector3f& init_t,
                        const PointCloud& input_cloud_src,
                        const PointCloud& input_cloud_tar,
                        const NormalCloud& input_normals_src,
                        const NormalCloud& input_normals_tar,
                        Eigen::Matrix3f& R,
                        Eigen::Vector3f& t);

private:

    float m_iss_salient_radius;
    float m_iss_non_max_radius;
    float m_iss_gamma_21;
    float m_iss_gamma_32;
    int m_iss_min_neighbors;
    int m_iss_threads;

    float m_harris3d_radius;
    float m_harris3d_nms_threshold;
    int m_harris3d_threads;
    bool m_harris3d_is_nms;
    bool m_harris3d_is_refine;

    float m_down_kp_voxel;

    float m_fpfh_feature_radius;

    float m_shot_feature_radius;

    uint32_t m_RANSAC_max_iter;
    float m_RANSAC_dist_threshold;
    float m_RANSAC_angle_threshold;
    float m_RANSAC_corres_rejection_rate;

    int m_ICP_normal_bins;
    size_t m_ICP_sampled_size;
    float m_ICP_max_corres_dist;
    size_t m_ICP_max_iter;
    float m_ICP_loss_epsilon;
};


#endif //HW9_REGISTRATION_HPP
