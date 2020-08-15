cmake_minimum_required(VERSION 3.15)
project(nano_vs_my)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)
include_directories(/usr/local/include/eigen3)

add_executable(nano_vs_my main.cpp include/utils.h include/KDTreeVectorOfVectorsAdaptor.h include/nanoflann.hpp include/kdtree.hpp include/resultSet.hpp)
#target_link_libraries(nano_vs_my ${OpenCV_LIBRARIES})






