cmake_minimum_required(VERSION 3.15)
project(pixienn)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(BLA_VENDOR OpenBLAS)

if (NOT DEFINED USE_CUDA OR USE_CUDA STREQUAL "")
    set(USE_CUDA 1)
endif ()

if (NOT DEFINED USE_CAIRO OR USE_CAIR STREQUAL "")
    set(USE_CAIRO 1)
endif ()

if (NOT DEFINED USE_PANGO OR USE_PANGO STREQUAL "")
    set(USE_PANGO 1)
endif ()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include_directories(${CMAKE_SOURCE_DIR}/cmake)

find_package(OpenBLAS REQUIRED)
find_package(Boost 1.74 REQUIRED)
find_package(GLIB REQUIRED COMPONENTS gobject)
find_package(HarfBuzz REQUIRED)
find_package(OpenCV 4.5.4 REQUIRED COMPONENTS core imgproc highgui)
find_package(TIFF REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Protobuf REQUIRED)

set(PROTOBUF_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tensorboard)
set(Protobuf_PROTOC_EXECUTABLE /usr/bin/protoc)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS --cpp_out=${PROTOBUF_OUTPUT_DIR} proto/*.proto)
SET_SOURCE_FILES_PROPERTIES(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

if (USE_CAIRO)
    find_package(Cairo 1.16.0)
    if (Cairo_FOUND)
        message(STATUS "CAIRO FOUND AND ENABLED")
    else ()
        message(STATUS "CAIRO NOT FOUND, WILL BE DISABLED")
    endif ()
else ()
    message(STATUS "CAIRO WILL BE DISABLED")
endif (USE_CAIRO)

if (USE_PANGO)
    find_package(Pango)
    if (PANGO_FOUND)
        find_package(Cairo 1.16.0 REQUIRED)
        message(STATUS "PANGO FOUND AND ENABLED")
    else ()
        message(STATUS "PANGO NOT FOUND, WILL BE DISABLED")
    endif ()
else ()
    message(STATUS "PANGO WILL BE DISABLED")
endif (USE_PANGO)

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

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++20 $<$<CONFIG:Debug>:-G>")
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
            src/ShortcutKernel.cu
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
        src/Box.cpp
        src/BurnInLRPolicy.cpp
        src/ColorMaps.cpp
        src/ConfusionMatrix.cpp
        src/ConnAlgo.cpp
        src/ConnLayer.cpp
        src/ConstantLRPolicy.cpp
        src/ConvAlgo.cpp
        src/ConvLayer.cpp
        src/CosineLRPolicy.cpp
        src/CpuUtil.cpp
        src/DetectAlgo.cpp
        src/DetectLayer.cpp
        src/Detection.cpp
        src/Error.cpp
        src/FileUtil.cpp
        src/Image.cpp
        src/ImageAugmenter.cpp
        src/ImageVec.cpp
        src/InvLRPolicy.cpp
        src/Layer.cpp
        src/Math.cpp
        src/MaxPoolAlgo.cpp
        src/MaxPoolLayer.cpp
        src/Model.cpp
        src/RegionLayer.cpp
        src/RouteLayer.cpp
        src/ShortcutAlgo.cpp
        src/ShortcutLayer.cpp
        src/SigmoidLRPolicy.cpp
        src/SmoothSteppedLRPolicy.cpp
        src/SteppedLRPolicy.cpp
        src/TiffIO.cpp
        src/Timer.cpp
        src/TrainBatch.cpp
        src/UpsampleAlgo.cpp
        src/UpsampleLayer.cpp
        src/Utility.cpp
        src/Validator.cpp
        src/YoloLayer.cpp
)

add_library(pixienn STATIC ${SOURCES})

set_target_properties(pixienn PROPERTIES OUTPUT_NAME "pixienn")

target_include_directories(pixienn PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CAIRO_INCLUDE_DIRS}
        ${Pango_INCLUDE_DIR}
        ${GLIB_INCLUDE_DIRS}
        ${HARFBUZZ_INCLUDE_DIR}
        ${OpenBLAS_INCLUDE_DIR}
        include)

if (Cairo_FOUND)
    target_compile_definitions(pixienn PRIVATE USE_CAIRO)
else ()
    remove_definitions(-DUSE_CAIRO)
endif ()

if (PANGO_FOUND)
    target_compile_definitions(pixienn PRIVATE USE_PANGO)
else ()
    remove_definitions(-DUSE_PANGO)
endif ()

if (USE_CUDA)
    target_include_directories(pixienn PRIVATE ${CUDNN_INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_definitions(pixienn PRIVATE USE_CUDA)
else ()
    remove_definitions(-DUSE_CUDA)
endif ()

include(PixiennLibs)
include(Pixienn)
include(PixiennTrain)

add_subdirectory(tests)
