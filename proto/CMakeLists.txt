file(GLOB PROTO_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
add_library(pixienn_proto ${PROTO_SRCS} ${PROTO_HDRS})
set_target_properties(pixienn_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(pixienn_proto ${PROTOBUF_LIBRARIES})
