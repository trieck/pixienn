
list(APPEND PIXIENN_LIBS
        pixienn
        tiff
        yaml-cpp
        ${OpenBLAS_LIB}
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
