add_executable(pixienn-bin src/Main.cpp)

if (USE_CUDA)
    target_compile_definitions(pixienn-bin PRIVATE -DUSE_CUDA)
    target_include_directories(pixienn-bin PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(pixienn-bin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(pixienn-bin PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif ()

set_target_properties(pixienn-bin PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(pixienn-bin PROPERTIES OUTPUT_NAME "pixienn")

target_include_directories(pixienn-bin PRIVATE include)

target_link_libraries(pixienn-bin ${PIXIENN_LIBS})
