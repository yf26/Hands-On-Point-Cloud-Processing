#include <chrono>
#include <cstdlib>
#include <nanoflann.hpp>
#include <random>
#include "spectralClustering.hpp"

#include "KDTreeVectorOfVectorsAdaptor.h"

#include "Spectra/GenEigsSolver.h"
#include "Spectra/MatOp/SparseGenMatProd.h"


typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double, std::milli> Duration;


std::vector<int> Spec_Cluster::fit(const my_matrix_t &points) {
    size_t  N = points.size();
    size_t  dim = points[0].size();

    Eigen::SparseMatrix<double> L(N, N);
    L.reserve(Eigen::VectorXi::Constant(N, K_neighbors));

    // TODO
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> features;
//    features.resize(N, -1);

    std::vector<int> belongings(N);

    auto build_start = Clock::now();
    buildKNNGraph(points, L);
    auto build_end = Clock::now();

    getNewFeatures(L, features);
    auto features_end = Clock::now();

    Duration build_time = build_end - build_start;
    Duration features_time = features_end - build_end;

    std::cout << "Build graph takes " << build_time.count() << " ms" << std::endl;
    std::cout << "Find features takes " << features_time.count() << " ms" << std::endl;

    KMeans(features, belongings);

//    auto t1 = Clock::now();
//    initial_choice(features);
//    Duration du = Clock::now() - t1;
//    std::cout << "Initialization takes " << du.count() << " ms" << std::endl;

    return belongings;
}

void Spec_Cluster::buildKNNGraph(const my_matrix_t& points, Eigen::SparseMatrix<double>& L)
{

    // build kdtree
//    auto build_start = Clock::now();
    size_t  N = points.size();
    typedef KDTreeVectorOfVectorsAdaptor<my_matrix_t, double> my_kdtree_t;
    size_t leaf_size = 4;
    my_kdtree_t mat_index(points[0].size(), points, leaf_size);
    mat_index.index->buildIndex();
//    auto build_end = Clock::now();

    // find knn for each point to build matrix W and D
    Eigen::SparseMatrix<double> W(N, N);
    L.reserve(Eigen::VectorXi::Constant(N, K_neighbors - 1));

    std::vector<std::pair<int, int>> index_pairs;
    for (int i = 0; i < N; i++)
    {
        std::vector<size_t> nn_indices(K_neighbors);
        std::vector<double> nn_dists_sqr(K_neighbors);
        nanoflann::KNNResultSet<double> result_set(K_neighbors);
        result_set.init(&nn_indices[0], &nn_dists_sqr[0]);
        mat_index.index->findNeighbors(result_set, &points[i][0], nanoflann::SearchParams());

        // TODO build weight matrix
        L.insert(i, i) = 1;

//        double row_sum = 0;
//        for (int idx = 0; idx < K_neighbors; idx++)
//        {
//            if (i == nn_indices[idx]) continue;
//            row_sum += 1 / sqrt(nn_dists_sqr[idx]);
//        }

        for (int idx = 0; idx < K_neighbors; idx++)
        {
            if (i == nn_indices[idx]) continue;
            W.insert(i, nn_indices[idx]) =  1 / sqrt(nn_dists_sqr[idx]);
            index_pairs.emplace_back(std::pair<int, int>(i, nn_indices[idx]));
        }
    }

//    for (auto& idx : index_pairs)
//    {
//        if (W.coeff(idx.first, idx.second) != W.coeff(idx.second, idx.first))
//            W.coeffRef(idx.first, idx.second) = 0;
//        std::cout << "Idx = " << idx.first << ", " << idx.second << std::endl;
//        std::cout << "val = " << W.coeff(idx.first, idx.second) << ", " << W.coeff(idx.second, idx.first) << std::endl;
//    }

//    std::cout << W << std::endl;

    assert(W.isCompressed());

    for (int i = 0; i < N; i++)
    {
        double row_sum = W.row(i).sum();
        W.row(i) /= row_sum;
    }


    L -= W;

    assert(L.isCompressed());

}


void Spec_Cluster::buildRNNGraph(const my_matrix_t& points, Eigen::SparseMatrix<double>& L)
{
    // build kdtree
//    auto build_start = Clock::now();
    typedef KDTreeVectorOfVectorsAdaptor<my_matrix_t, double> my_kdtree_t;
    size_t leaf_size = 4;
    my_kdtree_t mat_index(points[0].size(), points, leaf_size);
    mat_index.index->buildIndex();
//    auto build_end = Clock::now();

    // find knn for each point to build matrix W and D
    for (int i = 0; i < points.size(); i++)
    {
        std::vector<std::pair<size_t, double>> indices_dists;
//        nanoflann::SearchParams params;
        const size_t result_size = mat_index.index->radiusSearch(&points[i][0], search_radius, indices_dists, nanoflann::SearchParams());

        assert(result_size == indices_dists.size());

        // TODO build weight matrix
        L.insert(i, i) = 1;

        double row_sum = 0;
        for (int idx = 0; idx < result_size; idx++)
        {
            if (i == indices_dists[idx].first) continue;
            row_sum += 1 / sqrt(indices_dists[idx].second);
        }

        for (int idx = 0; idx < result_size; idx++)
        {
            if (i == indices_dists[idx].first) continue;
            L.insert(i, indices_dists[idx].first) =  - 1 / sqrt(indices_dists[idx].second) / row_sum;
        }
    }

    Eigen::SparseMatrix<double> LL;
    LL = 0.5 * (Eigen::SparseMatrix<double>(L.transpose()) + L);
    L = LL;
}


void Spec_Cluster::getNewFeatures(const Eigen::SparseMatrix<double> &L, Eigen::Matrix<double, -1, -1, Eigen::RowMajor>& features)
{
    using namespace Spectra;

//    auto eigen_start = Clock::now();
    SparseGenMatProd<double> op(L);

    GenEigsSolver<double, SMALLEST_REAL, SparseGenMatProd<double>> eigsSolver(&op, K_clusters_estimation, K_clusters_estimation+5);

    eigsSolver.init();
    int nconv = eigsSolver.compute(1e6, 1e-6, SMALLEST_REAL);
//    auto eigen_end = Clock::now();
//    Duration build_time = eigen_end - eigen_start;
//    std::cout << "Eigen decomposition takes " << build_time.count() << std::endl;


    if (eigsSolver.info() == SUCCESSFUL)
    {
        Eigen::VectorXd  eig_val = eigsSolver.eigenvalues().real();
        Eigen::MatrixXd  eig_vec = eigsSolver.eigenvectors().real();

        std::cout << "Eigenvalues found \n" << eig_val << std::endl;
//        std::cout << "Eigenvectors found \n" << eig_vec << std::endl;

        double diff = eig_val(1, 0) - eig_val(0, 0);
        for (int i = 1; i < eig_val.rows(); i++)
        {
            double temp = eig_val(i+1, 0) - eig_val(i, 0);
            if (temp > 50 * diff)
            {
                K_clusters = i + 1;
                break;
            }
        }

//        double eigengap = 0;
//        double idx = 0;
//        std::cout << "Eigengaps = " << std::endl;
//        for (int i = 1; i < eig_val.rows(); i++)
//        {
//            double diff = eig_val(i, 0) - eig_val(i-1, 0);
//            std::cout << diff << std::endl;
//            if (diff > eigengap)
//            {
//                eigengap = diff;
//                idx = i;
//            }
//        }
//        K_clusters = idx;

        std::cout << "Clusters number = " << K_clusters << std::endl;

        features = eig_vec.block(0, 0, eig_vec.rows(), K_clusters);

        std::cout << "Features got with size = " << features.rows() << ", " << features.cols() << std::endl;

//        Eigen::VectorXd ones(features.rows());
//        for (int i = 0; i < features.rows(); i++) ones(i) = 1;
//        Eigen::VectorXd test (features.rows());
//        test = eig_vec.col(K_clusters - 1);
//        std::cout << "Test the orthogonality " << ones.transpose() * test << std::endl;

    }
    else
        std::cout << "Eigenvalue decomposition failed!" << std::endl;


//    ///
//    std::ofstream output;
//    output.open("../features.txt");
//    for (int i = 0; i < features.rows(); i++)
//    {
//        for (int j = 0; j < features.cols(); j++)
//        {
//
//            output << features(i, j) << " ";
//        }
//        output << std::endl;
//    }
//    output.close();
//    ///

}


my_matrix_t initial_choice(const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>& features)
{

    size_t K = features.cols();
    size_t N = features.rows();
    std::vector<int> random_indices[K];

    my_matrix_t filtered_features;
    filtered_features.reserve(K);

    filtered_features.emplace_back(
        std::vector<double> (features.row(0).data(), features.row(0).data() + K)
    );

    for (int row = 1; row < N; row++)
    {
        if (filtered_features.size() == K)
            break;

        bool add = true;
        for (auto& item : filtered_features)
        {
            double diff = 0;
            for (int i = 0; i < K; i++)
                diff += pow(item[i] - features(row, i), 2);

            if ( diff < 0.0001 )
            {
                add = false;
                break;
            }
        }

        if (add)
            filtered_features.emplace_back(
                std::vector<double> (features.row(row).data(), features.row(row).data() + K)
            );
    }

    return filtered_features;
}


void Spec_Cluster::KMeans(const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>& features, std::vector<int>& belongings)
{
    size_t N = features.rows();
    size_t K = features.cols();
//    unsigned short random_indices[K];
//    random_indices[0] = rand() % N;
//    for (int i = 1; i < K; i++)
//    {
//        int temp = rand() % N;
//        while(temp == random_indices[i-1])
//            temp = rand() % N;
//        random_indices[i] = temp;
//    }
//
//    ///
//    random_indices[0] = 883;
//    random_indices[1] = 884;
//
//    std::cout << "Initial indices: \n";
//    for (auto& item : random_indices)
//        std::cout << item << " ";
//    std::cout << std::endl;
//    ///


    my_matrix_t centers = initial_choice(features);
//    centers.reserve(K);
//    for (const auto& idx : random_indices)
//    {
//
//        centers.emplace_back(
//            std::vector<double> (features.row(idx).data(), features.row(idx).data() + K)
//        );
//    }

    ///
    std::cout << "Initial centers: \n";
    for (auto& item : centers)
    {
        for (auto& num : item)
            std::cout << num << " ";
        std::cout << std::endl;
    }
    ///

    double tolerance = 0.0001;
    size_t max_iter = 200;
    size_t count = 0;

    while (true)
    {
        count++;


        typedef KDTreeVectorOfVectorsAdaptor<my_matrix_t, double> my_kdtree_t;
        my_kdtree_t mat_index(centers[0].size(), centers, 1);
        mat_index.index->buildIndex();

        my_matrix_t centers_new(K);
        for (int i = 0; i < K ; i++)
            centers_new[i] = std::vector<double> (K, 0);
        std::vector<int> count_cluster(K, 0);

        for (int row = 0; row < N; row++)
        {
            auto query = std::vector<double>(features.row(row).data(), features.row(row).data() + K);
            std::vector<size_t> nn_indices(1);
            std::vector<double> nn_dists_sqr(1);
            nanoflann::KNNResultSet<double> result_set(1);
            result_set.init(&nn_indices[0], &nn_dists_sqr[0]);
            mat_index.index->findNeighbors(result_set, &query[0], nanoflann::SearchParams());

            count_cluster[nn_indices[0]]++;
            for (int i = 0; i < K; i++)
            {
                centers_new[nn_indices[0]][i] += features(row, i);
            }

            belongings[row] = nn_indices[0];
        }

        for (int i = 0; i < K ; i++)
        {
            for (int j = 0; j < K ; j++)
            {
                centers_new[i][j] /= count_cluster[i];
            }
        }

        if (converged(centers_new, centers, tolerance) and count < max_iter)
        {
            centers = centers_new;
            std::cout << "Converged!" << std::endl;
            break;
        }
        centers = centers_new;
        if (count > max_iter)
        {
            std::cerr << "Maximal iteration number reached!" << std::endl;
            break;
        }

        if (count % 50 == 0)
        {
            std::cout << "Iteration " << count << std::endl;
            ///
            for (auto& item : centers_new)
            {
                for (auto& num : item)
                    std::cout << num << " ";
                std::cout << std::endl;
            }
            ///
        }

    }


}


bool converged(const my_matrix_t& A, const my_matrix_t& B, double threshold)
{
    int row = A.size();
    int col = A[0].size();
    for (int i = 0; i < row ; i++)
        for (int j = 0; j < col ; j++)
            if (fabs(A[i][j] - B[i][j]) < threshold)
                continue;
            else
            {
                return false;
            }

    return true;
}


my_matrix_t readPoints(const std::string& path)
{
    std::cout << "Read points from " << path << std::endl;

    std::vector<std::vector<double>> points;
    std::ifstream file(path);

    if (not file.good())
    {
        std::cerr << "Read file " << path << "failed!";
        exit(EXIT_FAILURE);
    }

    std::string line;

    while(getline(file, line))
    {
        std::vector<std::string> temp;
        boost::algorithm::split(temp, line, boost::is_any_of(","));
        points.emplace_back(std::vector<double>{{std::stod(temp[0]), std::stod(temp[1])}});
    }
    file.close();

    std::cout << "Points dim = " << points.size() << ", " << points[0].size() << std::endl;
    return points;
}

