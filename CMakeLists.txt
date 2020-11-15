cmake_minimum_required(VERSION 3.15)
project(pixienn)

set(CMAKE_CXX_STANDARD 17)

set(BLA_VENDOR OpenBLAS)

find_package(BLAS REQUIRED)
find_package(CUDA REQUIRED)
find_package(HDF5 REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TIFF REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(xtl REQUIRED)
find_package(yaml-cpp REQUIRED)

set(SOURCES
        include/Utility.h
        include/Singleton.h
        include/SingletonRegistry.h
        src/BatchNormLayer.cpp
        src/ConnLayer.cpp
        src/ConvLayer.cpp
        src/Darknet.cpp
        src/DetectLayer.cpp
        src/Error.cpp
        src/Image.cpp
        src/Utility.cpp
        src/Layer.cpp
        src/MaxPoolLayer.cpp
        src/Model.cpp
        src/main.cpp
        )

set(INCLUDE_DIRS include)
include_directories(${INCLUDE_DIRS})

add_executable(pixienn ${SOURCES})

target_link_libraries(pixienn ${OpenCV_LIBS} yaml-cpp xtensor xtensor::optimize tiff openblas)


