cmake_minimum_required(VERSION 3.15)
project(homework1)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(/usr/local/include/eigen3/)

add_executable (homework1 main.cpp)
target_link_libraries(homework1 ${PCL_LIBRARIES})
