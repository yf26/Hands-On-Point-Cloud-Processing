//
// Created by yu on 30.04.20.
//

#ifndef HW3_SPECTRALCLUSTERING_HPP
#define HW3_SPECTRALCLUSTERING_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include <vector>
#include <boost/algorithm/string.hpp>

typedef std::vector<std::vector<double>> my_matrix_t;

class Spec_Cluster
{
private:
    size_t K_clusters;
    size_t K_neighbors;
    double search_radius;
    size_t K_clusters_estimation;

public:
    Spec_Cluster(int k_neigh, size_t k_clus_estimation)
    {
        K_neighbors = k_neigh;
        K_clusters = 1;
        K_clusters_estimation = k_clus_estimation;
        search_radius = 0;
    }

    Spec_Cluster(double r, size_t k_clus_estimation)
    {
        search_radius = r;
        K_clusters = 1;
        K_clusters_estimation = k_clus_estimation;
        K_neighbors = 0;
    }

    std::vector<int> fit(const my_matrix_t& points);

private:
    void buildKNNGraph(const my_matrix_t& points, Eigen::SparseMatrix<double>& L);
    void buildRNNGraph(const my_matrix_t& points, Eigen::SparseMatrix<double>& L);
    void getNewFeatures(const Eigen::SparseMatrix<double>& L, Eigen::Matrix<double, -1, -1, Eigen::RowMajor>& features);
    void KMeans(const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>& features, std::vector<int>& belongings);
};


std::vector<std::vector<double>> readPoints(const std::string& path);

bool converged(const my_matrix_t& A, const my_matrix_t& B, double threshold);

my_matrix_t initial_choice(const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>& features);

#endif //HW3_SPECTRALCLUSTERING_HPP
