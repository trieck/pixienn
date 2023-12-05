
list(APPEND PIXIENN_LIBS
        pixienn
        tiff
        yaml-cpp
        ${BLAS_openblas_LIBRARY}
        boost_filesystem
        boost_program_options
        ${CUDART_LIBRARY}
        ${CUBLAS_LIBRARY}
        ${CUDNN_LIBRARY}
        ${GLIB_LIBRARIES}
        ${GLIB_GOBJECT_LIBRARIES}
        ${HARFBUZZ_LIBRARY}
        ${OpenCV_LIBS}
)

if (USE_PANGO)
    list(APPEND PIXIENN_LIBS
            ${CAIRO_LIBRARIES}
            ${Pango_LIBRARIES}
    )

endif (USE_PANGO)
