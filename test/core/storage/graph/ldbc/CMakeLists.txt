if ( CTEST_ALL OR CTEST_STORAGE )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/storage/graph/ldbc/ldbc_csr_graph_test_app )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/storage/graph/ldbc/ldbc_adj_graph_test_app )
    
    add_executable( ldbc_csr_graph_test_app ldbc_csr_graph_test.cpp)
    add_executable( ldbc_adj_graph_test_app ldbc_adj_graph_test.cpp)
    target_link_libraries(ldbc_adj_graph_test_app PRIVATE "-ldl" stdc++fs)
    target_link_libraries(ldbc_csr_graph_test_app PRIVATE "-ldl" stdc++fs)

    add_test( ldbc_csr_graph_test ldbc_csr_graph_test_app )
    add_test( ldbc_adj_graph_test ldbc_adj_graph_test_app )
endif()