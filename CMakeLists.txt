cmake_minimum_required(VERSION 3.15)
project(pixienn)

set(CMAKE_CXX_STANDARD 17)

set(BLA_VENDOR OpenBLAS)

find_package(BLAS REQUIRED)
find_package(Boost REQUIRED)
find_package(CUDA REQUIRED)
find_package(HDF5 REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TIFF REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtl REQUIRED)
find_package(yaml-cpp REQUIRED)

file(GLOB SOURCES "src/*.cpp" "include/*.h")

add_executable(pixienn ${SOURCES})

target_include_directories(pixienn PRIVATE include)
target_link_libraries(pixienn ${OpenCV_LIBS} yaml-cpp xtensor xtensor::optimize tiff openblas)


