if ( BUILD_ALL OR BUILD_MICROBMS )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/src/microbenchmarks/graph/vertex_storage_benchmark_app )
    
    add_executable( vertex_storage_benchmark_app vertex_storage_benchmark.cpp)
endif()