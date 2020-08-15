cmake_minimum_required(VERSION 3.15)
project(hw7)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(/usr/include/eigen3)
include_directories(include)

file(GLOB LIB_SOURCES src/*.cpp)
file(GLOB LIB_HEADERS include/*.cpp)
add_library(lib_iss_detector ${LIB_SOURCES} ${LIB_HEADERS})

add_executable(hw7 main.cpp ${INCLUDE} ${SOURCE})
target_link_libraries(hw7 ${PCL_LIBRARIES} lib_iss_detector)