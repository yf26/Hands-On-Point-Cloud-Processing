//
// Created by yu on 29.05.20.
//

#ifndef HW7_ISS_DETECTOR_HPP
#define HW7_ISS_DETECTOR_HPP

#include <cstdint>
#include <vector>
#include "kdtree.hpp"
#include "Eigen/Eigenvalues"

typedef std::vector<std::vector<float>> MyPCDType;

class ISSKeypoint
{
public:
    void useWeightedCovMat(bool use);
    void setLocalRadius(float r);
    void setNonMaxRadius(float r);
    void setThreshold(float g21, float g32);
    void setMinNeighbors(int n);
    void setInputPointCloud(MyPCDType& input_point_cloud);
    void compute(MyPCDType& keypoints);

private:
    Eigen::Vector3f getEigenvalues(size_t i);

private:
    bool use_w_cov_mat;
    float local_radius;
    float non_max_radius;
    float gama21, gama32;
    int min_neighbors;
    MyPCDType point_cloud;
    std::vector<std::vector<int>> rnn_idx;
};

#endif //HW7_ISS_DETECTOR_HPP
