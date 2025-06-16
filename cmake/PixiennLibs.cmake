
list(APPEND PIXIENN_LIBS
        pixienn
        tiff
        yaml-cpp
        ${OpenBLAS_LIB}
        boost_filesystem
        boost_program_options
        CUDA::cudart
        CUDA::cublas
        ${CUDNN_LIBRARY}
        ${GLIB_LIBRARIES}
        ${GLIB_GOBJECT_LIBRARIES}
        ${HARFBUZZ_LIBRARY}
        ${OpenCV_LIBS}
        pixienn_proto
        protobuf::libprotobuf
)

if (Cairo_FOUND)
    list(APPEND PIXIENN_LIBS
            ${CAIRO_LIBRARIES}
    )
endif (Cairo_FOUND)

if (PANGO_FOUND)
    list(APPEND PIXIENN_LIBS
            ${CAIRO_LIBRARIES}
            ${Pango_LIBRARIES}
    )

endif (PANGO_FOUND)
