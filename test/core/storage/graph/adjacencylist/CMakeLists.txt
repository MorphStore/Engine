if ( CTEST_ALL OR CTEST_STORAGE )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/storage/graph/adjacencylist/ldbc_graph_adjacencylist_test_app )

    add_executable( ldbc_graph_adjacencylist_test_app ldbc_graph_adjacencylist.cpp)
    target_compile_options( ldbc_graph_adjacencylist_test_app PRIVATE
            -Werror
            -Wall
            -Wextra
            -pedantic
            -fstack-protector-all
            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( ldbc_graph_adjacencylist_test_app PRIVATE "-ldl" stdc++fs)

    add_test( ldbc_graph_adjacency_test ldbc_graph_adjacencylist_test_app )
endif()