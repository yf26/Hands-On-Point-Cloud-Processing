#include <Eigen/Core>

#include "spectralClustering.hpp"

#include <iostream>
#include <fstream>
#include <random>

int main()
{
    std::vector<std::string> files = {
        "aniso.txt",
        "blobs.txt",
        "circle.txt",
        "moons.txt",
        "varied.txt",
    };

    for (auto& file : files)
    {
        std::cout << "********* Data set: " << file << " *********" << std::endl;
        std::string path = "../data/" + file;
        auto points = readPoints(path);

        Spec_Cluster test_sc(10, 8);

        auto result = test_sc.fit(points);

        std::ofstream output;
        std::string output_file = "../result/predict_" + file;

        output.open(output_file);
        for (auto& item : result)
        {
            output << item << std::endl;
        }
        output.close();
        std::cout << "Done!\n" << std::endl;
    }







    return 0;
}
