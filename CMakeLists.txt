cmake_minimum_required(VERSION 3.15)
project(pixienn)

set(CMAKE_CXX_STANDARD 17)
set(BLA_VENDOR OpenBLAS)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(BLAS REQUIRED)
find_package(Boost 1.69 EXACT REQUIRED)
find_package(CUDA 10.0 REQUIRED)
find_package(CUDNN REQUIRED)
find_package(HDF5 REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenCV 3.4.10 EXACT REQUIRED)
find_package(TIFF REQUIRED)
find_package(nlohmann_json 3.9.1 EXACT REQUIRED)
find_package(xtensor 0.21.9 EXACT REQUIRED)
find_package(xtl 0.6.21 EXACT REQUIRED)
find_package(yaml-cpp REQUIRED)

set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})
enable_language("CUDA")

set(OUTPUTFILE ${CMAKE_CURRENT_BINARY_DIR}cuda_script)
set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/src/check_cuda/check_cuda.cu)

execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -lcuda ${CUDAFILE} -o ${OUTPUTFILE} RESULT_VARIABLE CUDA_RETURN_CODE)

if (NOT ${CUDA_RETURN_CODE} EQUAL 0)
    message(FATAL_ERROR "Cannot compile CUDA source file \"${CUDAFILE}\".")
endif ()

execute_process(COMMAND ${OUTPUTFILE} RESULT_VARIABLE CUDA_RETURN_CODE OUTPUT_VARIABLE ARCH)
if (${CUDA_RETURN_CODE} EQUAL 0)
    set(CUDA_NVCC_FLAGS "${ARCH} -std=c++14")
    message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
else ()
    message(FATAL_ERROR "Cannot execute CUDA program.")
endif ()

file(GLOB SOURCES "src/*.cpp" "src/*.cu" "include/*.h" "include/*.cuh")

cuda_add_executable(pixienn ${SOURCES})

set_target_properties(pixienn PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(pixienn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(pixienn PRIVATE include)
target_link_directories(pixienn PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

target_link_libraries(pixienn
        ${CUDNN_LIBRARY}
        ${OpenCV_LIBS}
        boost_chrono
        boost_filesystem
        boost_program_options
        boost_thread
        nlohmann_json::nlohmann_json
        openblas
        tiff
        xtensor
        xtensor::optimize
        yaml-cpp
        )

