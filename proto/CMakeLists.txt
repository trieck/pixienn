file(GLOB PROTO_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
add_library(pixienn_proto ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(pixienn_proto ${PROTOBUF_LIBRARIES})
