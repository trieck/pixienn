cmake_minimum_required(VERSION 3.15)
project(pixienn)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
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

    file(GLOB CUDA_HEADERS "include/*.cuh")
    list(APPEND SOURCES
            ${CUDA_HEADERS}
            src/ActivationKernels.cu
            src/BiasKernels.cu
            src/CudaError.cpp
            src/CudaUtils.cu
            src/PoolKernels.cu
            src/UpsampleKernels.cu
    )
    message(STATUS "CUDA FOUND AND ENABLED")
else ()
    message(STATUS "CUDA WILL BE DISABLED")
endif (USE_CUDA)

file(GLOB HEADERS "include/*.h")
list(APPEND SOURCES
        ${HEADERS}
        src/Activation.cpp
        src/BatchNormAlgo.cpp
        src/BatchNormLayer.cpp
        src/ConnLayer.cpp
        src/ConvAlgo.cpp
        src/ConvLayer.cpp
        src/DetectLayer.cpp
        src/Detection.cpp
        src/Error.cpp
        src/Image.cpp
        src/Layer.cpp
        src/Math.cpp
        src/MaxPoolLayer.cpp
        src/Model.cpp
        src/NMS.cpp
        src/RouteLayer.cpp
        src/Timer.cpp
        src/UpsampleLayer.cpp
        src/Utility.cpp
        src/YoloLayer.cpp
)

add_library(pixienn STATIC ${SOURCES})
set_target_properties(pixienn PROPERTIES OUTPUT_NAME "pixienn")

target_include_directories(pixienn PRIVATE ${OpenCV_INCLUDE_DIRS} include)

if (USE_CUDA)
    target_include_directories(pixienn PRIVATE ${CUDNN_INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_definitions(pixienn PRIVATE -DUSE_CUDA)
endif ()

add_executable(pixienn-bin src/main.cpp)

if (USE_CUDA)
    target_compile_definitions(pixienn-bin PRIVATE -DUSE_CUDA)
    target_include_directories(pixienn-bin PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(pixienn-bin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(pixienn-bin PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif ()

set_target_properties(pixienn-bin PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(pixienn-bin PROPERTIES OUTPUT_NAME "pixienn")

target_include_directories(pixienn-bin PRIVATE include)

target_link_libraries(pixienn-bin
        ${OpenCV_LIBS}
        pixienn
        tiff
        yaml-cpp
        ${BLAS_openblas_LIBRARY}
        boost_filesystem
        boost_program_options
        ${CUDART_LIBRARY}
        ${CUBLAS_LIBRARY}
        ${CUDNN_LIBRARY}
)

add_subdirectory(tests)

