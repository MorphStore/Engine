if ( CTEST_ALL OR CTEST_OPERATORS )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/graph/simple/bfs_simple_csr_graph_test_app )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/graph/simple/bfs_simple_adj_graph_test_app )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/graph/simple/page_rank_simple_csr_graph_test_app )
    
    add_executable( bfs_simple_csr_graph_test_app bfs_simple_csr_graph_test.cpp)
    add_executable( bfs_simple_adj_graph_test_app bfs_simple_adj_graph_test.cpp)
    add_executable( page_rank_simple_csr_graph_test_app page_rank_simple_csr_graph_test.cpp)

    add_test( bfs_simple_csr_graph_test_app bfs_simple_csr_graph_test_app )
    add_test( bfs_simple_adj_graph_test_app bfs_simple_adj_graph_test_app )
    add_test( page_rank_simple_csr_graph_test_app page_rank_simple_csr_graph_test_app )
endif()