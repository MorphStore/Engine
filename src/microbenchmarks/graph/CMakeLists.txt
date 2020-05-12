if ( BUILD_ALL OR BUILD_MICROBMS )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/src/microbenchmarks/graph/vertex_storage_benchmark_app )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/src/microbenchmarks/graph/edge_storage_benchmark_app )
    
    add_executable( vertex_storage_benchmark_app vertex_storage_benchmark.cpp)
    add_executable( edge_storage_benchmark_app edge_storage_benchmark.cpp)
endif()