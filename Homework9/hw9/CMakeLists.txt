cmake_minimum_required(VERSION 3.15)
project(hw9)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL 1.8 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
        include/
        ${PCL_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB SOURCES src/*.cpp)
file(GLOB HEADERS include/*.hpp include/*.h)

add_executable(hw9 main.cpp ${SOURCES} ${HEADERS})
target_link_libraries(hw9 ${PCL_LIBRARIES})

add_executable(test test.cpp ${SOURCES} ${HEADERS})
target_link_libraries(test ${PCL_LIBRARIES})