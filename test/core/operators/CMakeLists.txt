if ( CTEST_ALL OR CTEST_OPERATORS )
	MESSAGE( "Building test, TESTING OPERATORS " )
	add_executable( agg_sum_group_based_test_app agg_sum_group_based_test.cpp )
	add_executable( agg_sum_whole_column_test_app agg_sum_whole_column_test.cpp )
	add_executable( group_binary_test_app group_binary_test.cpp )
	add_executable( group_unary_test_app group_unary_test.cpp )
	add_executable( intersect_test_app intersect_test.cpp )
	add_executable( join_test_app join_test.cpp )
	add_executable( project_test_app project_test.cpp )
	add_executable( select_test_app select_test.cpp )
	add_executable( operation_lookup_test_app lookup_test.cpp )
	add_executable( operation_summation_test_app summation_test.cpp )
	
	target_compile_options( agg_sum_group_based_test_app PRIVATE
	        -Werror
	        -Wall
	        -Wextra
	        -pedantic
	        -fstack-protector-all
	        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( agg_sum_group_based_test_app PRIVATE "-ldl" )
	
	target_compile_options( agg_sum_whole_column_test_app PRIVATE
	        -Werror
	        -Wall
	        -Wextra
	        -pedantic
	        -fstack-protector-all
	        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( agg_sum_whole_column_test_app PRIVATE "-ldl" )
	
	target_compile_options( group_binary_test_app PRIVATE
	        -Werror
	        -Wall
	        -Wextra
	        -pedantic
	        -fstack-protector-all
	        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( group_binary_test_app PRIVATE "-ldl" )
	
	target_compile_options( group_unary_test_app PRIVATE
	        -Werror
	        -Wall
	        -Wextra
	        -pedantic
	        -fstack-protector-all
	        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( group_unary_test_app PRIVATE "-ldl" )
	
	target_compile_options( intersect_test_app PRIVATE
	        -Werror
	        -Wall
	        -Wextra
	        -pedantic
	        -fstack-protector-all
	        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( intersect_test_app PRIVATE "-ldl" )
	
	target_compile_options( join_test_app PRIVATE
	        -Werror
	        -Wall
	        -Wextra
	        -pedantic
	        -fstack-protector-all
	        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( join_test_app PRIVATE "-ldl" )
	
	target_compile_options( project_test_app PRIVATE
	        -Werror
	        -Wall
	        -Wextra
	        -pedantic
	        -fstack-protector-all
	        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( project_test_app PRIVATE "-ldl" )
	
	target_compile_options( select_test_app PRIVATE
	        -Werror
	        -Wall
	        -Wextra
	        -pedantic
	        -fstack-protector-all
	        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( select_test_app PRIVATE "-ldl" )
	
	target_compile_options( operation_lookup_test_app PRIVATE
	        -Werror
	        -Wall
	        -Wextra
	        -pedantic
	        -fstack-protector-all
	        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( operation_lookup_test_app PRIVATE "-ldl" )
	
	
	target_compile_options( operation_summation_test_app PRIVATE
	        -Werror
	        -Wall
	        -Wextra
	        -pedantic
	        -fstack-protector-all
	        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( operation_summation_test_app PRIVATE "-ldl" )
	
	add_test( agg_sum_group_based_test agg_sum_group_based_test_app )
	add_test( agg_sum_whole_column_test agg_sum_whole_column_test_app )
	add_test( group_binary_test group_binary_test_app )
	add_test( group_unary_test group_unary_test_app )
	add_test( intersect_test intersect_test_app )
	add_test( join_test join_test_app )
	add_test( project_test project_test_app )
	add_test( select_test select_test_app )
	add_test( operation_lookup_test operation_lookup_test_app )
	add_test( operation_summation_test operation_summation_test_app )
endif()
	