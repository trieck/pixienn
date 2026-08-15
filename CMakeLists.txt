cmake_minimum_required(VERSION 3.15)
project(pixienn)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(BLA_VENDOR OpenBLAS)

option(PIXIENN_ENABLE_WARNINGS
        "Enable the project warning policy"
        ON)
option(PIXIENN_ENABLE_CLANG_TIDY
        "Run clang-tidy during C++ compilation"
        OFF)
option(PIXIENN_ENABLE_CLANG_FORMAT
        "Provide the clang-format-check target"
        ON)

if (PIXIENN_ENABLE_WARNINGS)
    add_compile_options(
            $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall>
            $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wextra>
            $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Werror>
            $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wno-system-headers>)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
            CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
        add_compile_options(-Wno-deprecated-enum-enum-conversion)
    endif ()
endif ()

if (PIXIENN_ENABLE_CLANG_TIDY)
    find_program(PIXIENN_CLANG_TIDY_EXECUTABLE NAMES clang-tidy clang-tidy-18 REQUIRED)
    set(CMAKE_CXX_CLANG_TIDY
            "${PIXIENN_CLANG_TIDY_EXECUTABLE};--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy"
            CACHE STRING "clang-tidy command" FORCE)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "clang-tidy with GCC requires the GCC compile database; use a Clang compiler for CMake-integrated tidy runs")
    endif ()
endif ()

option(PIXIENN_BUILD_PYTHON_BINDINGS
        "Build the optional Python interface"
        OFF)

if (PIXIENN_BUILD_PYTHON_BINDINGS)
    include(FetchContent)

    # Keep the binding dependency explicit and reproducible. FetchContent uses
    # CMake's normal download/cache machinery, so no developer-local Python or
    # pybind11 installation path is required.
    set(PYBIND11_FINDPYTHON ON CACHE BOOL
            "Use modern CMake Python discovery for pybind11" FORCE)
    FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
            GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif ()

if (POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif ()

if (NOT DEFINED USE_CUDA OR USE_CUDA STREQUAL "")
    set(USE_CUDA 1)
endif ()

if (NOT DEFINED USE_CAIRO OR USE_CAIRO STREQUAL "")
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
    set(CMAKE_CUDA_STANDARD 20)

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
    find_package(CUDAToolkit REQUIRED)
    find_package(CUDNN 8 REQUIRED)

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++20 $<$<CONFIG:Debug>:-G>")

    file(GLOB CUDA_HEADERS "include/*.cuh")
    list(APPEND SOURCES
            ${CUDA_HEADERS}
            src/Activation.cu
            src/Adam.cu
            src/BiasKernels.cu
            src/CenterNetKernels.cu
            src/CudaError.cpp
            src/CudaUtils.cu
            src/Gemm.cpp
            src/MaxPoolKernels.cu
            src/ShortcutKernel.cu
            src/SoftmaxKernel.cu
            src/UpsampleKernels.cu
            src/YoloKernels.cu
    )
    message(STATUS "CUDA FOUND AND ENABLED")
else ()
    message(STATUS "CUDA WILL BE DISABLED")
endif (USE_CUDA)

file(GLOB HEADERS "include/*.h")
list(APPEND SOURCES
        ${HEADERS}
        src/BatchLoader.cpp
        src/BatchNorm.cpp
        src/Box.cpp
        src/BurnInLRPolicy.cpp
        src/CRC32.cpp
        src/CenterNetTargetBuilder.cpp
        src/YoloTargetBuilder.cpp
        src/ColorMaps.cpp
        src/ConfusionMatrix.cpp
        src/ConstantLRPolicy.cpp
        src/CosineLRPolicy.cpp
        src/CpuUtil.cpp
        src/DarkBox.cpp
        src/Detection.cpp
        src/Error.cpp
        src/FileUtil.cpp
        src/Image.cpp
        src/ImageAugmenter.cpp
        src/ImageVec.cpp
        src/InvLRPolicy.cpp
        src/Math.cpp
        src/MiniBatch.cpp
        src/Model.cpp
        src/RandomLRPolicy.cpp
        src/RecordWriter.cpp
        src/SigmoidLRPolicy.cpp
        src/SmoothCyclicDecayLRPolicy.cpp
        src/SmoothSteppedLRPolicy.cpp
        src/SteppedLRPolicy.cpp
        src/TiffIO.cpp
        src/Timer.cpp
        src/Utility.cpp
)

add_library(pixienn STATIC ${SOURCES})
set_target_properties(pixienn PROPERTIES OUTPUT_NAME "pixienn")

if (PIXIENN_ENABLE_CLANG_FORMAT)
    find_program(PIXIENN_CLANG_FORMAT_EXECUTABLE NAMES clang-format clang-format-18)
    if (PIXIENN_CLANG_FORMAT_EXECUTABLE)
        file(GLOB_RECURSE PIXIENN_FORMAT_FILES CONFIGURE_DEPENDS
                "${CMAKE_SOURCE_DIR}/include/*.h"
                "${CMAKE_SOURCE_DIR}/src/*.cpp"
                "${CMAKE_SOURCE_DIR}/tests/src/*.cpp")
        add_custom_target(pixienn-format-check
                COMMAND ${PIXIENN_CLANG_FORMAT_EXECUTABLE}
                --dry-run --Werror --style=file ${PIXIENN_FORMAT_FILES}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                COMMENT "Checking C++ formatting")
    else ()
        message(STATUS "clang-format not found; pixienn-format-check is unavailable")
    endif ()
endif ()

add_subdirectory(proto)
add_dependencies(pixienn pixienn_proto)

list(APPEND INCLUDE_DIRS
        ${CMAKE_CURRENT_BINARY_DIR}/proto
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${CAIRO_INCLUDE_DIRS}
        ${Pango_INCLUDE_DIR}
        ${GLIB_INCLUDE_DIRS}
        ${HARFBUZZ_INCLUDE_DIR}
        ${OpenBLAS_INCLUDE_DIR}
)

target_include_directories(pixienn PRIVATE ${INCLUDE_DIRS})

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

option(PIXIENN_BUILD_TESTS "Build the native C++ test suite" ON)
if (PIXIENN_BUILD_TESTS)
  add_subdirectory(tests)
endif ()

if (PIXIENN_BUILD_PYTHON_BINDINGS)
  set_target_properties(pixienn PROPERTIES POSITION_INDEPENDENT_CODE ON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  pybind11_add_module(pixienn_native MODULE src/python/pixienn_module.cpp)
  set_target_properties(pixienn_native PROPERTIES
          OUTPUT_NAME _native
          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/pixienn")
  target_link_libraries(pixienn_native PRIVATE ${PIXIENN_LIBS})
  target_include_directories(pixienn_native PRIVATE ${INCLUDE_DIRS})
  if (USE_CUDA)
    target_compile_definitions(pixienn_native PRIVATE USE_CUDA)
    target_include_directories(pixienn_native PRIVATE ${CUDNN_INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  endif ()
  add_custom_command(TARGET pixienn_native POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_directory
                  ${CMAKE_CURRENT_SOURCE_DIR}/python/pixienn
                  ${CMAKE_BINARY_DIR}/python/pixienn)
  add_custom_target(pixienn-python-test
          COMMAND ${CMAKE_COMMAND} -E copy_directory
                  ${CMAKE_CURRENT_SOURCE_DIR}/python/pixienn
                  ${CMAKE_BINARY_DIR}/python/pixienn
          COMMAND ${CMAKE_COMMAND} -E env
                  "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
                  ${Python3_EXECUTABLE} -m unittest discover
                  -s ${CMAKE_CURRENT_SOURCE_DIR}/python/tests
                  -p test_*.py
          DEPENDS pixienn_native
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  install(TARGETS pixienn_native
          LIBRARY DESTINATION pixienn)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/pixienn/
          DESTINATION pixienn
          FILES_MATCHING
          PATTERN "*.py"
          PATTERN "__pycache__" EXCLUDE)
endif ()
