#[[
if ( CTEST_ALL OR CTEST_VECTOR )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/vector_test_app )
	add_executable( vector_test_app main.cpp )

    target_compile_options( vector_test_app PRIVATE
#            -Werror
#            -Wall
#            -Wextra
#            -pedantic
            -fstack-protector-all
            -msse4.2 -msse2
           # -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl 
           
            $<$<CONFIG:DEBUG>:-DDEBUG> )
  
    target_link_libraries( vector_test_app PRIVATE "-ldl" )
    
    add_test( vector_test vector_test_app )
endif()]]
#[[
if ( CTEST_ALL OR CTEST_VECTOR )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/agg_sum_app )
    add_executable( agg_sum_app operators/agg_sum_whole_column_test.cpp )

    target_compile_options( agg_sum_app PRIVATE
                        -Wall
                        -Wextra
                        -pedantic
            -fstack-protector-all
            -mavx2
            # -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl

            $<$<CONFIG:DEBUG>:-DDEBUG> )

    target_link_libraries( agg_sum_app PRIVATE "-ldl" )

    add_test( agg_sum_vectorized_test agg_sum_app )
endif()]]
#[[if ( CTEST_ALL OR CTEST_VECTOR )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/hash_set_app )
    add_executable( hash_set_app operators/hash_set_test.cpp )

    target_compile_options( hash_set_app PRIVATE
            -Wall
            -Wextra
            -pedantic
            -fstack-protector-all
            -mavx2
            # -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl

            $<$<CONFIG:DEBUG>:-DDEBUG> )

    target_link_libraries( hash_set_app PRIVATE "-ldl" )

    add_test( hash_set_vectorized_test hash_set_app )
endif()]]
if ( CTEST_ALL OR CTEST_VECTOR )
    FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/semi_join_app )
    add_executable( semi_join_app operators/semi_join_test.cpp )

    target_compile_options( semi_join_app PRIVATE
            -Wall
            -Wextra
            -pedantic
            -fstack-protector-all
            -mavx2
            # -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl

            $<$<CONFIG:DEBUG>:-DDEBUG> )

    target_link_libraries( semi_join_app PRIVATE "-ldl" )

    add_test( semi_join_vectorized_test semi_join_app )
endif()