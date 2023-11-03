cmake_minimum_required(VERSION 3.15)
project(pixienn)

set(CMAKE_CXX_STANDARD 17)
set(BLA_VENDOR OpenBLAS)

set(USE_CUDA 1)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(BLAS REQUIRED)
find_package(Boost 1.74 REQUIRED)
find_package(HDF5 REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenCV 4.5.4 REQUIRED)
find_package(TIFF REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)
find_package(xtensor 0.23.10 REQUIRED)
find_package(xtl 0.7.2 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CUDA 10.0)

if (USE_CUDA AND CUDA_FOUND)
    find_package(CUDNN 8 REQUIRED)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif ()
    set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})
    set(OUTPUTFILE ${CMAKE_CURRENT_BINARY_DIR}/cuda_script)
    set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/src/check_cuda/check_cuda.cu)
    enable_language(CUDA)
    execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -lcuda ${CUDAFILE} -o ${OUTPUTFILE} RESULT_VARIABLE CUDA_RETURN_CODE)
    if (NOT ${CUDA_RETURN_CODE} EQUAL 0)
        message(FATAL_ERROR "Cannot compile CUDA source file \"${CUDAFILE}\".")
    endif ()

    execute_process(COMMAND ${OUTPUTFILE} RESULT_VARIABLE CUDA_RETURN_CODE OUTPUT_VARIABLE ARCH)
    if (${CUDA_RETURN_CODE} EQUAL 0)
        set(CUDA_NVCC_FLAGS "${ARCH} -std=c++17 $<$<CONFIG:Debug>:-G>")
        message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
    else ()
        message(FATAL_ERROR "Cannot execute CUDA program.")
    endif ()

    file(GLOB CUDA_SOURCES "src/*.cu" "include/*.cuh")
    list(APPEND SOURCES ${CUDA_SOURCES})

    message(STATUS "CUDA FOUND AND ENABLED")
else ()
    message(STATUS "CUDA WILL BE DISABLED")
endif (USE_CUDA AND CUDA_FOUND)

file(GLOB CPP_SOURCES "src/*.cpp" "include/*.h")
list(APPEND SOURCES ${CPP_SOURCES})

if (USE_CUDA AND CUDA_FOUND)
    cuda_add_executable(pixienn ${SOURCES})
    cuda_add_cublas_to_target(pixienn)
    target_compile_definitions(pixienn PRIVATE -DUSE_CUDA)
    set_target_properties(pixienn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_directories(pixienn PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
else ()
    add_executable(pixienn ${SOURCES})
endif ()

set_target_properties(pixienn PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pixienn PRIVATE include)

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
