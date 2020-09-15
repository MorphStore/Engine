if ( CTEST_ALL OR CTEST_STORAGE )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/storage/graph/simple/simple_csr_graph_test_app )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/storage/graph/simple/simple_adj_graph_test_app )
    
    add_executable( simple_csr_graph_test_app simple_csr_graph_test.cpp)
    add_executable( simple_adj_graph_test_app simple_adj_graph_test.cpp)

    add_test( simple_csr_graph_test simple_csr_graph_test_app )
    add_test( simple_adj_graph_test simple_adj_graph_test_app )
endif()