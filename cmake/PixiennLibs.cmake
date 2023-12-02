
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
    ${CAIRO_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GLIB_GOBJECT_LIBRARIES}
    ${HARFBUZZ_LIBRARY}
    ${Pango_LIBRARIES}
    ${OpenCV_LIBS}
)
