#ifndef HW2_TEST_HPP
#define HW2_TEST_HPP
#include "resultSet.hpp"
#include "bst.hpp"
#include "kdtree.hpp"
#include "octree.hpp"
#include <fstream>
#include <random>
#include <chrono>

std::vector<std::vector<double>>
readBinary(const std::string& fileName)
{
    std::fstream input(fileName.c_str(), std::ios::in | std::ios::binary);
    input.seekg(0, std::ios::beg);
    std::vector<std::vector<double>> points;

    if (not input.good())
    {
        std::cerr << "Read file " << fileName << "failed!";
        exit(EXIT_FAILURE);
    }

    for (int i = 0; input.good() and !input.eof(); i++)
    {
        float temp[4];
        input.read((char* )&temp, 4*sizeof(float));
        points.emplace_back(std::vector<double>{temp[0], temp[1], temp[2]});
    }
    input.close();
    std::cout << points.size() << " points read!" << std::endl;
    return points;
}

void testBST()
{

    int dataSize = 100000;
    double queryKey  = 4.2;

    TreeNode<double>* root;
    root = nullptr;
    std::vector<double> data;
    data.resize(dataSize);

    // initialize test data set
    std::iota(data.begin(), data.end(), 0);

    std::shuffle(data.begin(), data.end(), std::mt19937(std::random_device()()));

    std::cout << "Test data set: " << std::endl;
    for (auto& item : data)
    {
        std::cout << item << " ";
    }
    std::cout << std::endl;


    // build binary search tree
    for (int i = 0; i < dataSize; i++)
    {
        insert(root, data[i], i);
    }
//    double queryKey = 4.2;
    std::cout << "Query Key = " << queryKey << std::endl;


    std::cout << "*********** Test searchIteratively and searchRecursively *************" << std::endl;
    auto positionPtr1 = searchIteratively(root, queryKey);
    if (positionPtr1 != nullptr)
        std::cout << "At searched position, key = " << positionPtr1->key << std::endl;
    else
        std::cout << "QueryKey " << queryKey << " not found!" << std::endl;
    std::cout << std::endl;



//    std::cout << "*********** Test KNNResultSet *************" << std::endl;
//    KNNResultSet testResultSet(3);
//    testResultSet.addPoint(0, 0);
//    testResultSet.addPoint(1, 1);
//    testResultSet.addPoint(2, 2);
//    testResultSet.addPoint(1, 3);
//    testResultSet.addPoint(0, 4);
//    testResultSet.list();

    std::cout << "*********** Test KNNSearch *************" << std::endl;
    int K = 3;
    std::cout << "K = " << K << std::endl;
    auto start1 = std::chrono::high_resolution_clock::now();
    KNNResultSet knnResultSet(K);
    KNNSearch(root, knnResultSet, queryKey);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration1 = end1 - start1;
    std::cout << "KNN search takes " << duration1.count() << "ms" << std::endl;

    std::cout << "Comparision times " << knnResultSet.comparisionCount << std::endl;

    knnResultSet.list();
    std::cout << std::endl;

    std::cout << "*********** Test RadiusNNSearch *************" << std::endl;
    double radius = 1.21;
    std::cout << "Radius = " << radius << std::endl;
    auto start2 = std::chrono::high_resolution_clock::now();
    RadiusNNResultSet rnnResultSet(radius);
    RadiusNNSearch(root, rnnResultSet, queryKey);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration2 = end2 - start2;
    std::cout << "RadiusNN search takes " << duration2.count() << "ms" << std::endl;

    std::cout << "Comparision times " << rnnResultSet.comparisionCount << std::endl;

    rnnResultSet.list();


    deleteTree(root);
}

std::vector<std::vector<double>> getPCD()
{
    std::cout << "************** TEST KDTREE*******************" << std::endl;
    auto read_start = std::chrono::high_resolution_clock::now();
    auto db = readBinary("../000000.bin");
    auto read_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> read_time = read_end - read_start;
    std::cout << "Read file time " << read_time.count()<< "ms" << std::endl;
    return db;
}


std::vector<std::vector<double>>
generateRandomPointCloud(const size_t N, const size_t dim, const double max_range = 10.0)
{
    std::vector<std::vector<double>> samples;
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

    std::ofstream out_file;
    out_file.open("../test.txt");
    out_file << samples.size() << std::endl;
    for (auto& item : samples)
    {
        for (int i = 0; i < dim; i++)
            out_file << item[i] << " ";
        out_file << std::endl;
    }
    out_file.close();

    return samples;
}

#define query_idx 1000
void testKDTree()
{
//    auto db = getPCD();
//    auto db_test = std::vector<std::vector<double>>(db.begin(), db.begin() + 100000);
//    srand(static_cast<unsigned int>(time(nullptr)));
    auto db_test = generateRandomPointCloud(100000, 10, 20);

    ///
    std::cout << "Test point cloud size " << db_test.size() << ", " << db_test[0].size() << std::endl;
    ///

//    auto query = std::vector<double>{db[5][0]+0.1, db[5][1]+0.2, db[5][2]-0.1};
    auto query = db_test[query_idx];

    int K = 8;
    int leaf_size = 3;
    KNNResultSet result_set(K);

    auto start = std::chrono::high_resolution_clock::now();
    Node* root = KDTreeConstruction(db_test, leaf_size);

    auto build = std::chrono::high_resolution_clock::now();
    KDTreeKNNSearch(root, db_test, result_set, query);
    if (root->isLeaf())
        std::cout << "Error!!!" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> build_time = build - start;
    std::chrono::duration<double, std::milli> search_time = end - build;

    std::cout << "Build KDTree time " << build_time.count() << "ms" << std::endl;
    std::cout << "KDTree depth = " << TreeDepth(root) << std::endl;
    std::cout << "------ KNN search --------" << std::endl;
    std::cout << "KDTree KNN search time " << search_time.count() << "ms" << std::endl;
    std::cout << "Total time " << build_time.count()+search_time.count() << "ms" << std::endl;
    std::cout << "Comparison times = " << result_set.comparisionCount << std::endl;
    result_set.list();

    std::cout << "------ RNN search --------" << std::endl;
    double radius = 1.299;

    RadiusNNResultSet result_set_rnn(radius);

    auto start_rnn = std::chrono::high_resolution_clock::now();
    KDTreeRadiusNNSearch(root, db_test, result_set_rnn, query);
    auto end_rnn = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> rnn_search_time = end_rnn - start_rnn;

    std::cout << "KDTree RNN search time " << rnn_search_time.count() << "ms" << std::endl;
    std::cout << "Total time " << build_time.count()+rnn_search_time.count() << "ms" << std::endl;
    std::cout << "Comparison times = " << result_set_rnn.comparisionCount << std::endl;
    result_set_rnn.list();

    std::cout << "------ Destruction of KDTree --------" << std::endl;
    auto destruction_start = std::chrono::high_resolution_clock::now();
    KDTreeDestruction();
    auto destruction_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> destruction_time = destruction_end - destruction_start;
    std::cout << "Destruction time " << destruction_time.count()<< "ms" << std::endl;
    std::cout << std::endl;
}


void testOctree()
{
//    auto db = getPCD();
//    auto db_test = std::vector<std::vector<double>>(db.begin(), db.begin() + 100000);
    auto db_test = generateRandomPointCloud(100000, 3, 20);

//    auto query = std::vector<double>{db[5][0]+0.1, db[5][1]+0.2, db[5][2]-0.1};
    auto query = db_test[query_idx];
    int leaf_size = 32;
    double min_extent = 0.0001;

    int K = 8;
    KNNResultSet result_set_knn(K);

    auto start = std::chrono::high_resolution_clock::now();
    Octant* root = OctreeConstruction(db_test, leaf_size, min_extent);
    auto build = std::chrono::high_resolution_clock::now();

    OctreeKNNSearch(root, db_test, result_set_knn, query);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> build_time = build - start;
    std::chrono::duration<double, std::milli> search_time = end - build;

    std::cout << "Build Octree time " << build_time.count()<< "ms" << std::endl;
    std::cout << "------ KNN search --------" << std::endl;
    std::cout << "Octree KNN search time " << search_time.count()<< "ms" << std::endl;
    std::cout << "Total time " << build_time.count() + search_time.count() << "ms" << std::endl;
    std::cout << "Comparison times = " << result_set_knn.comparisionCount << std::endl;
    result_set_knn.list();

    std::cout << "------ RNN search --------" << std::endl;
    double radius = 0.5;

    RadiusNNResultSet result_set_rnn(radius);

    auto start_rnn = std::chrono::high_resolution_clock::now();
    OctreeRadiusNNSearch(root, db_test, result_set_rnn, query);
    auto end_rnn = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> rnn_search_time = end_rnn - start_rnn;

    std::cout << "Octree RNN search time " << rnn_search_time.count()<< "ms" << std::endl;
    std::cout << "Comparison times = " << result_set_rnn.comparisionCount << std::endl;
    std::cout << "Total time " << build_time.count() + rnn_search_time.count() << "ms" << std::endl;
    result_set_rnn.list();
    std::cout << "------ Destruction of Octree --------" << std::endl;
    auto destruction_start = std::chrono::high_resolution_clock::now();
    OctreeDestruction();
    auto destruction_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> destruction_time = destruction_end - destruction_start;
    std::cout << "Destruction time " << destruction_time.count()<< "ms" << std::endl;
}

#endif //HW2_TEST_HPP