cmake_minimum_required(VERSION 3.15)
project(pixienn)

set(CMAKE_CXX_STANDARD 17)
set(BLA_VENDOR OpenBLAS)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(BLAS_LIBRARIES "/opt/homebrew/opt/openblas/lib/libblas.dylib" CACHE STRING "BLAS library path")
set(BLAS_INCLUDE_DIRS "/opt/homebrew/opt/openblas/include" CACHE PATH "BLAS include directory path")
set(LAPACK_LIBRARIES "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" CACHE STRING "LAPACK library path")
set(OpenCV_DIR "/opt/homebrew/opt/opencv/share/OpenCV")
set(Boost_LIBRARIES "/opt/homebrew/opt/boost/lib")
set(Boost_INCLUDE_DIR "/opt/homebrew/opt/boost/include")
set(TIFF_LIBRARY "/opt/homebrew/opt/libtiff/lib")
set(TIFF_INCLUDE_DIR "/opt/homebrew/opt/libtiff/include")
set(nlohmann_json_DIR "/opt/homebrew/opt/nlohmann-json/lib/cmake/nlohmann-json")
set(xtensor_DIR "/opt/homebrew/opt/xtensor/lib/cmake/xtensor")
set(yaml-cpp_DIR "/opt/homebrew/opt/yaml-cpp/lib/cmake/yaml-cpp")

include_directories("/opt/homebrew/opt/openblas/include")

find_package(BLAS REQUIRED)
find_package(Boost REQUIRED COMPONENTS chrono filesystem program_options)
find_package(LAPACK REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TIFF REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtl REQUIRED)
find_package(yaml-cpp REQUIRED)


file(GLOB SOURCES "src/*.cpp" "src/*.cu" "include/*.h" "include/*.cuh")

link_directories(/opt/homebrew/opt/openblas/lib/)
link_directories(/opt/homebrew/opt/libtiff/lib/)
link_directories(/opt/homebrew/opt/yaml-cpp/lib/)


add_executable(pixienn ${SOURCES})
target_link_libraries(pixienn Boost::filesystem Boost::program_options)

set_target_properties(pixienn PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(pixienn PRIVATE include)

target_link_libraries(pixienn
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${TIFF_LIBRARY}
        ${YAML_CPP_LIBRARIES}
        nlohmann_json::nlohmann_json
        openblas
        tiff
        xtensor
        xtensor::optimize
        yaml-cpp
)

