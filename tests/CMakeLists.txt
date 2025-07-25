cmake_minimum_required(VERSION 3.15)

project(pixienn-tests)

if (COVERAGE)
    list(APPEND CMAKE_CXX_FLAGS "--coverage")
    set(CODE_COVERAGE ON)
endif ()

find_package(GTest REQUIRED)
find_package(GMock REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Protobuf REQUIRED)

if (NOT DEFINED Python3_EXECUTABLE)
    message(FATAL_ERROR "Python3_EXECUTABLE not set! Please pass -DPython3_EXECUTABLE=/path/to/python.")
endif ()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")

file(GLOB TEST_HEADERS "include/*.h")

list(APPEND TEST_SOURCES
        ${HEADERS}
        src/activations.cpp
        src/augment.cpp
        src/blas.cpp
        src/box.cpp
        src/confuse.cpp
        src/cosine.cpp
        src/crc32.cpp
        src/darkbox.cpp
        src/imagebatch.cpp
        src/main.cpp
        src/math.cpp
        src/model.cpp
        src/sigmoid.cpp
        src/smoothcycle.cpp
        src/smoothsteps.cpp
        src/steppedlr.cpp
        src/tensor.cpp
        src/tiff.cpp
        src/vector.cpp
)

add_executable(pixienn-test ${TEST_SOURCES})

target_include_directories(pixienn-test PRIVATE
        "${pixienn_SOURCE_DIR}/include"
        "include"
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_BINARY_DIR}/proto
)

if (USE_CUDA)
    target_compile_definitions(pixienn-test PRIVATE -DUSE_CUDA)
    target_include_directories(pixienn-test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(pixienn-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(pixienn-test PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif ()

set_target_properties(pixienn-test PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(pixienn-test
        ${PIXIENN_LIBS}
        GTest::gtest_main
        GTest::gmock_main
        ${HDF5_CXX_LIBRARIES}
)

set(SPEC_FILE ${CMAKE_SOURCE_DIR}/python/specs/model_test.yaml)
set(TESTS_H5_PATH ${CMAKE_CURRENT_BINARY_DIR}/model_test.h5)
set(PYTHON_SCRIPT ${CMAKE_SOURCE_DIR}/python/compile_model_spec.py)

add_definitions(-DTESTS_H5_PATH="${TESTS_H5_PATH}")

add_custom_command(
        OUTPUT ${TESTS_H5_PATH}
        COMMAND ${Python3_EXECUTABLE} ${PYTHON_SCRIPT}
        ${SPEC_FILE} ${TESTS_H5_PATH}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
        ${PYTHON_SCRIPT}
        ${SPEC_FILE}
        COMMENT "Generating ${TESTS_H5_PATH} from ${SPEC_FILE}"
)

enable_testing()
include(GoogleTest)

gtest_discover_tests(pixienn-test)

add_custom_target(generate_test_data DEPENDS ${TESTS_H5_PATH})
add_custom_target(test COMMAND ${CMAKE_CTEST_COMMAND})

add_dependencies(test pixienn-test generate_test_data)
add_dependencies(pixienn-test generate_test_data)
