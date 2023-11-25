cmake_minimum_required(VERSION 3.15)

project(pixienn_tests)

find_package(GTest REQUIRED)
find_package(GMock REQUIRED)

add_executable(pixienn_test
        batchnormal.cpp
        convolve.cpp
        main.cpp
        math.cpp
        tensor.cpp
        vector.cpp
)

target_include_directories(pixienn_test PRIVATE "${pixienn_SOURCE_DIR}/include")

if (USE_CUDA)
    target_compile_definitions(pixienn_test PRIVATE -DUSE_CUDA)
    target_include_directories(pixienn_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(pixienn_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(pixienn_test PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif ()

set_target_properties(pixienn_test PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(pixienn_test
        ${OpenCV_LIBS}
        pixienn
        tiff
        yaml-cpp
        ${BLAS_openblas_LIBRARY}
        boost_filesystem
        boost_program_options
        GTest::gtest_main
        GTest::gmock_main
        ${CUDART_LIBRARY}
        ${CUBLAS_LIBRARY}
        ${CUDNN_LIBRARY}
)

enable_testing()
include(GoogleTest)

gtest_discover_tests(pixienn_test)

add_custom_target(test COMMAND ${CMAKE_CTEST_COMMAND})
add_dependencies(test pixienn_test)
