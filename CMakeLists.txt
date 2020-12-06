cmake_minimum_required(VERSION 3.15)
project(pixienn)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(BLA_VENDOR OpenBLAS)

find_package(BLAS REQUIRED)
find_package(Boost REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDNN REQUIRED)
find_package(HDF5 REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TIFF REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtl REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

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


file(GLOB SOURCES "src/*.cpp" "src/*.cu" "include/*.h")

cuda_add_executable(pixienn ${SOURCES})

set_target_properties(pixienn PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(pixienn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(pixienn PRIVATE include)
target_link_directories(pixienn PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

target_link_libraries(pixienn ${OpenCV_LIBS} yaml-cpp xtensor xtensor::optimize tiff openblas boost_program_options
        boost_filesystem nlohmann_json::nlohmann_json)

