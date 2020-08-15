cmake_minimum_required(VERSION 3.15)
project(hw3)

set(CMAKE_CXX_STANDARD 14)


include_directories(include)
include_directories(/usr/local/include/eigen3)


add_executable(hw3 main.cpp include/utils.h include/KDTreeVectorOfVectorsAdaptor.h include/nanoflann.hpp spectralClustering.cpp include/spectralClustering.hpp)







