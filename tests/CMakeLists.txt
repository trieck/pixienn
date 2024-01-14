cmake_minimum_required(VERSION 3.15)

project(pixienn-tests)

if (COVERAGE)
    list(APPEND CMAKE_CXX_FLAGS "--coverage")
    set(CODE_COVERAGE ON)
endif ()

find_package(GTest REQUIRED)
find_package(GMock REQUIRED)

add_executable(pixienn-test
        activations.cpp
        augment.cpp
#        batchnorm.cpp
        blas.cpp
        confuse.cpp
#        connected.cpp
#        convolve.cpp
        cosine.cpp
        darkbox.cpp
#        detect.cpp
        imagebatch.cpp
        main.cpp
        math.cpp
#        maxpool.cpp
        sigmoid.cpp
        smoothsteps.cpp
        steppedlr.cpp
        tensor.cpp
        tiff.cpp
#        upsample.cpp
        vector.cpp
)

target_include_directories(pixienn-test PRIVATE "${pixienn_SOURCE_DIR}/include")

if (USE_CUDA)
    target_compile_definitions(pixienn-test PRIVATE -DUSE_CUDA)
    target_include_directories(pixienn-test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(pixienn-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(pixienn-test PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif ()

set_target_properties(pixienn-test PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(pixienn-test ${PIXIENN_LIBS} GTest::gtest_main GTest::gmock_main)

enable_testing()
include(GoogleTest)

gtest_discover_tests(pixienn-test)

add_custom_target(test COMMAND ${CMAKE_CTEST_COMMAND})
add_dependencies(test pixienn-test)
