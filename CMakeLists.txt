# project settings
cmake_minimum_required(VERSION 3.12)

project(ExaMPM LANGUAGES CXX VERSION 0.1.0)

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
find_package(MPI REQUIRED)
find_package(Kokkos 3 REQUIRED)
find_package(Cabana REQUIRED)
find_package(Cajita REQUIRED)
# TODO: Have libbat find TBB/threads in its package config
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)
find_package(libbat REQUIRED)

# library
add_subdirectory(src)

# examples
add_subdirectory(examples)
