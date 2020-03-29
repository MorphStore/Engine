if ( CTEST_ALL OR CTEST_OPERATORS )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/graph/ldbc/bfs_ldbc_csr_graph_test_app )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/graph/ldbc/bfs_ldbc_adj_graph_test_app )
    
    add_executable( bfs_ldbc_csr_graph_test_app bfs_ldbc_csr_graph_test.cpp)
    add_executable( bfs_ldbc_adj_graph_test_app bfs_ldbc_adj_graph_test.cpp)
    target_link_libraries(bfs_ldbc_csr_graph_test_app PRIVATE "-ldl" stdc++fs)
    target_link_libraries(bfs_ldbc_adj_graph_test_app PRIVATE "-ldl" stdc++fs)

    add_test( bfs_ldbc_csr_graph_test_app bfs_ldbc_csr_graph_test_app )
    add_test( bfs_ldbc_adj_graph_test_app bfs_ldbc_adj_graph_test_app )
endif()