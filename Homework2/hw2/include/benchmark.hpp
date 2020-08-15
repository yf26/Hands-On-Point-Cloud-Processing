
#ifndef HW2_BENCHMARK_HPP
#define HW2_BENCHMARK_HPP

#include "test.hpp"
void benchmark()
{
    auto db_raw = readBinary("../000000.bin");
    auto db = std::vector<std::vector<double>>(db_raw.begin(), db_raw.begin() + 10000);

    int leaf_size = 1;
    double min_extent = 0.0001;
    int k = 8;
    double radius = 1;
    int iteration_num = 10000;

    typedef std::chrono::duration<double, std::milli> Duration;
    typedef std::chrono::steady_clock Clock;



    std::cout << "octree --------------" << std::endl;
    auto knn_time_sum_oc = (Duration)0.;
    auto radius_time_sum_oc = (Duration)0.;

    auto begin_oc = Clock::now();
    Octant* root_oc = OctreeConstruction(db, leaf_size, min_extent);
    Duration construction_time_sum_oc = Clock::now() - begin_oc;

    for (int i = 0; i < 10000; i++)
    {
        auto query = db[i];

        auto begin_t = Clock::now();
        KNNResultSet result_set_knn(k);
        OctreeKNNSearch(root_oc, db, result_set_knn, query);
        knn_time_sum_oc += Clock::now() - begin_t;

        begin_t = Clock::now();
        RadiusNNResultSet result_set_rnn(k);
        OctreeRadiusNNSearch(root_oc, db, result_set_rnn, query);
        radius_time_sum_oc += Clock::now() - begin_t;
    }
    OctreeDestruction();
    printf("Octree: build %7.4fms, knn %7.4fms, radius %7.4fms",
           construction_time_sum_oc.count(),
           knn_time_sum_oc.count() / iteration_num,
           radius_time_sum_oc.count() / iteration_num);

    std::cout << std::endl;
    std::cout << "kdtree --------------" << std::endl;
    auto knn_time_sum_kd = (Duration)0.;
    auto radius_time_sum_kd = (Duration)0.;

    auto begin_kd = Clock::now();
    Node* root_kd = KDTreeConstruction(db, leaf_size);
    Duration construction_time_sum_kd = Clock::now() - begin_kd;

    for (int i = 0; i < 10000; i++)
    {
        auto query = db[i];

        auto begin_t = Clock::now();
        KNNResultSet result_set_knn(k);
        KDTreeKNNSearch(root_kd, db, result_set_knn, query);
        knn_time_sum_kd += Clock::now() - begin_t;

        begin_t = Clock::now();
        RadiusNNResultSet result_set_rnn(k);
        KDTreeRadiusNNSearch(root_kd, db, result_set_rnn, query);
        radius_time_sum_kd += Clock::now() - begin_t;
    }
    KDTreeDestruction();
    printf("Kdtree: build %7.4fms, knn %7.4fms, radius %7.4fms",
           construction_time_sum_kd.count(),
           knn_time_sum_kd.count() / iteration_num,
           radius_time_sum_kd.count() / iteration_num);
}


#endif //HW2_BENCHMARK_HPP
