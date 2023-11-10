cmake_minimum_required(VERSION 3.15)
project(pixienn)

set(CMAKE_CXX_STANDARD 20)
set(BLA_VENDOR OpenBLAS)

if (NOT DEFINED USE_CUDA OR USE_CUDA STREQUAL "")
    set(USE_CUDA 1)
endif ()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(BLAS REQUIRED)
find_package(Boost 1.74 REQUIRED)
find_package(OpenCV 4.5.4 REQUIRED)
find_package(TIFF REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)
find_package(xtensor 0.23.10 REQUIRED)
find_package(xtl 0.7.2 REQUIRED)
find_package(yaml-cpp REQUIRED)

if (USE_CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "75")
    endif ()

    if (NOT DEFINED CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER STREQUAL "")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER_ENV_VAR "/usr/local/cuda/bin/nvcc")
        set(CUDA_NVCC_FLAGS_DEBUG "")
        set(CUDA_PROPAGATE_HOST_FLAGS CACHE BOOL ON)
    endif ()

    enable_language(CUDA)
    find_package(CUDNN 8 REQUIRED)

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 $<$<CONFIG:Debug>:-G>")
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUBLAS_LIBRARY cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

    file(GLOB CUDA_SOURCES "src/*.cu" "include/*.cuh")
    list(APPEND SOURCES ${CUDA_SOURCES})
    message(STATUS "CUDA FOUND AND ENABLED")
else ()
    message(STATUS "CUDA WILL BE DISABLED")
endif (USE_CUDA)

file(GLOB CPP_SOURCES "src/*.cpp" "include/*.h")
list(APPEND SOURCES ${CPP_SOURCES})

add_executable(pixienn ${SOURCES})

if (USE_CUDA)
    target_compile_definitions(pixienn PRIVATE -DUSE_CUDA)
    target_include_directories(pixienn PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(pixienn ${CUDART_LIBRARY} ${CUBLAS_LIBRARY})
    set_target_properties(pixienn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(pixienn PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif ()

set_target_properties(pixienn PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pixienn PRIVATE include)

target_link_libraries(pixienn
        ${BLAS_openblas_LIBRARY}
        ${CUDNN_LIBRARY}
        ${OpenCV_LIBS}
        boost_chrono
        boost_filesystem
        boost_program_options
        boost_thread
        nlohmann_json::nlohmann_json
        tiff
        yaml-cpp
)
