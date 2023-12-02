add_executable(pixienn-train src/Train.cpp)

if (USE_CUDA)
    target_compile_definitions(pixienn-train PRIVATE -DUSE_CUDA)
    target_include_directories(pixienn-train PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(pixienn-train PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(pixienn-train PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif ()

set_target_properties(pixienn-train PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(pixienn-train PRIVATE include)

target_link_libraries(pixienn-train ${PIXIENN_LIBS})
