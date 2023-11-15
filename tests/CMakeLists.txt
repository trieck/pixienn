cmake_minimum_required(VERSION 3.15)

project(pixienn_tests)

find_package(GTest REQUIRED)

add_executable(tensor_test main.cpp tensor.cpp)
target_include_directories(tensor_test PRIVATE "${pixienn_SOURCE_DIR}/include")

if (USE_CUDA)
    target_compile_definitions(tensor_test PRIVATE -DUSE_CUDA)
    target_include_directories(tensor_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(tensor_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(tensor_test PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif ()

set_target_properties(tensor_test PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(tensor_test
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
        pixienn
        tiff
        yaml-cpp
        ${BLAS_openblas_LIBRARY}
        boost_chrono
        boost_filesystem
        boost_program_options
        GTest::gtest_main
        ${CUDART_LIBRARY}
        ${CUBLAS_LIBRARY}
        ${CUDNN_LIBRARY}
)

enable_testing()
include(GoogleTest)
gtest_discover_tests(tensor_test)
