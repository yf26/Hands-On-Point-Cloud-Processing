cmake_minimum_required(VERSION 2.8.12)
project(mylib)

include_directories(/usr/local/include/eigen3)
add_subdirectory(pybind11)
pybind11_add_module(mylib src/mylib.cpp)
