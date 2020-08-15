//
// Created by yu on 29.05.20.
//

#include "iss_detector.hpp"

void ISSKeypoint::useWeightedCovMat(bool use)
{
    use_w_cov_mat = use;
}

void ISSKeypoint::setLocalRadius(float r)
{
    local_radius = r;
}

void ISSKeypoint::setNonMaxRadius(float r)
{
    non_max_radius = r;
}

void ISSKeypoint::setThreshold(float g21, float g32)
{
    gama21 = g21;
    gama32 = g32;
}

void ISSKeypoint::setMinNeighbors(int n)
{
    min_neighbors = n;
}

void ISSKeypoint::setInputPointCloud(MyPCDType& input_point_cloud)
{
    point_cloud = input_point_cloud;
}

void ISSKeypoint::compute(MyPCDType& keypoints)
{
    keypoints.clear();

    // precompute the radius neighbors indices for each point in the input pcd
    size_t num_points = point_cloud.size();
    rnn_idx.resize(num_points);

    Node* root = KDTreeConstruction(point_cloud, 12);

    for (int i = 0; i < num_points; i++)
    {
        RadiusNNResultSet result_set(local_radius);
        KDTreeRadiusNNSearch(root, point_cloud, result_set, point_cloud[i]);

        rnn_idx[i].reserve(result_set.size());
        for (const auto& dist_idx : result_set.distIndexList)
            rnn_idx[i].emplace_back(dist_idx.index);
    }

//    ///
//    std::cout << "rnn_idx = " << std::endl;
//    for (int i = 0; i < num_points; i++)
//    {
//        for (auto& item : rnn_idx[i])
//            std::cout << item << " ";
//        std::cout << std::endl;
//    }
//    ///

    std::vector<float> lambda3_vec_forall(num_points, -1);
    for (int i = 0; i < num_points; i++)
    {
       if (rnn_idx[i].size() >= 3)
       {
           Eigen::Vector3f eigenvalues = getEigenvalues(i);
           float lambda1 = eigenvalues[2];
           float lambda2 = eigenvalues[1];
           float lambda3 = eigenvalues[0];

           if (lambda2 / lambda1 < gama21 and lambda3 / lambda2 < gama32 and lambda3 > 0)
               lambda3_vec_forall[i] = lambda3;
       }
    }



    // non max filter
    for (int i = 0; i < num_points; i++)
    {
        if (lambda3_vec_forall[i] == -1) continue;

        RadiusNNResultSet result_set(non_max_radius);
        KDTreeRadiusNNSearch(root, point_cloud, result_set, point_cloud[i]);
        if (result_set.size() < min_neighbors) continue;

        bool is_keypoint = true;
        for (const auto& dist_idx : result_set.distIndexList)
            if (lambda3_vec_forall[i] < lambda3_vec_forall[dist_idx.index])
            {
                is_keypoint = false;
                break;
            }

        if (is_keypoint)
            keypoints.emplace_back(point_cloud[i]);

    }

    KDTreeDestruction();
}


Eigen::Vector3f ISSKeypoint::getEigenvalues(size_t i)
{

    // MyPCDType to eigen matrix
    size_t num_neighbors = rnn_idx[i].size();

    Eigen::Vector3f center = Eigen::Vector3f{point_cloud[i][0], point_cloud[i][1], point_cloud[i][2]};


    Eigen::Matrix3f cov_matrix = Eigen::Matrix3f::Zero(3, 3);

    if (use_w_cov_mat)
    {
        float weight_nn;
        float weight_sum = 0;
        for (int j = 0; j < num_neighbors; j++)
        {
            size_t idx_neighbor = rnn_idx[i][j];
            weight_nn = 1.f / rnn_idx[idx_neighbor].size();
            weight_sum += weight_nn;
            Eigen::Vector3f neighbor = Eigen::Vector3f{
                point_cloud[idx_neighbor][0], point_cloud[idx_neighbor][1], point_cloud[idx_neighbor][2]};
            cov_matrix += weight_nn * (neighbor - center) * (neighbor - center).transpose();
        }
        cov_matrix /= weight_sum;
    }
    else
    {
        for (int j = 0; j < num_neighbors; j++)
        {
            size_t idx_neighbor = rnn_idx[i][j];
            Eigen::Vector3f neighbor = Eigen::Vector3f{
                point_cloud[idx_neighbor][0], point_cloud[idx_neighbor][1], point_cloud[idx_neighbor][2]};
            cov_matrix += (neighbor - center) * (neighbor - center).transpose();
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(cov_matrix);
    Eigen::Vector3f eigenvalues = eigensolver.eigenvalues().real();
//    std::cout << eigenvalues.transpose() << std::endl;
    return eigenvalues; // from lowest to highest
}
