//#include <Eigen/Core>
#include <nanoflann.hpp>
#include "KDTreeVectorOfVectorsAdaptor.h"

#include "kdtree.hpp"
#include "resultSet.hpp"

//#include <cstdlib>
#include <cstdlib>
#include <iostream>
#include <chrono>



typedef std::vector<std::vector<double> > my_vector_of_vectors_t;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double, std::milli> Duration;

void generateRandomPointCloud(my_vector_of_vectors_t &samples, const size_t N, const size_t dim, const double max_range = 10.0)
{
    std::cout << "Generating "<< N << " random points...";
    samples.resize(N);
    for (size_t i = 0; i < N; i++)
    {
        samples[i].resize(dim);
        for (size_t d = 0; d < dim; d++)
            samples[i][d] = max_range * (rand() % 1000) / (1000.0);
    }
    std::cout << "done\n";
    std::cout << "Point cloud size: " << samples.size() << ", " << samples[0].size() << std::endl;

//    for (auto& item : samples)
//    {
//        std::cout << item[0] << ", " << item[1] << ", " << item[2] << std::endl;
//    }

}

void kdtree_demo(const size_t nSamples, const size_t dim)
{
    my_vector_of_vectors_t  samples;

    const double max_range = 50;

    // Generate points:
    generateRandomPointCloud(samples, nSamples, dim, max_range);


    // Query point:
    std::vector<double> query_pt(dim);
    for (size_t d = 0;d < dim; d++)
        query_pt[d] = max_range * (rand() % 1000) / (1000.0);

    // other params:
    const size_t leaf_size = 10;
    const size_t num_results = 8;

    std::cout << "********** NANOFLANN KDTREE *************" << std::endl;
    // construct a kd-tree index:
    // Dimensionality set at run-time (default: L2)
    // ------------------------------------------------------------
    typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;

    auto build_start = Clock::now();
    my_kd_tree_t   mat_index(dim /*dim*/, samples, leaf_size /* max leaf */ );
    mat_index.index->buildIndex();
    auto build_end = Clock::now();

    // do a knn search
    std::vector<size_t>   ret_indexes(num_results);
    std::vector<double> out_dists_sqr(num_results);

    nanoflann::KNNResultSet<double> resultSet(num_results);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
    mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
    auto search_end = Clock::now();

    Duration build_time = build_end - build_start;
    Duration search_time = search_end - build_end;

    std::cout << "knnSearch(nn="<<num_results<<"): \n";
    for (size_t i = 0; i < num_results; i++)
        std::cout << "ret_index["<<i<<"]=" << ret_indexes[i] << " out_dist_sqr=" << out_dists_sqr[i] << std::endl;

    std::cout << "Build takes " << build_time.count() << " ms" << std::endl;
    std::cout << "Search takes " << search_time.count() << " ms" << std::endl;


    std::cout << "********** MY KDTREE *************" << std::endl;
    auto my_build_start = Clock::now();
    Node* root = KDTreeConstruction(samples, leaf_size);
    auto my_build_end = Clock::now();
    KNNResultSet result_set(num_results);
    KDTreeKNNSearch(root, samples, result_set, query_pt);
    auto my_search_end = Clock::now();
    Duration my_build_time = my_build_end - my_build_start;
    Duration my_search_time = my_search_end - my_build_end;
    result_set.list();
    std::cout << "Build takes " << my_build_time.count() << " ms" << std::endl;
    std::cout << "Search takes " << my_search_time.count() << " ms" << std::endl;
    std::cout << "comparisionCount " << result_set.comparisionCount << std::endl;
    std::cout << "Tree depth " << TreeDepth(root) << std::endl;
}



int main()
{
    // Randomize Seed
    srand(static_cast<unsigned int>(time(nullptr)));
    kdtree_demo(100000 /* samples */, 3 /* dim */);
}
