if ( CTEST_ALL OR CTEST_OPERATORS )
	MESSAGE( "Building test, TESTING OPERATORS " )
	add_executable( agg_sum_test_app agg_sum_test.cpp )
	add_executable( group_test_app group_test.cpp )
	add_executable( join_test_app join_test.cpp )
	add_executable( project_test_app project_test.cpp )
	add_executable( select_test_app select_test.cpp )
	add_executable( operation_lookup_test_app lookup_test.cpp )
	add_executable( operation_summation_test_app summation_test.cpp )
	
	target_compile_options( agg_sum_test_app PRIVATE
	        -Werror
	        -Wall
	        -Wextra
	        -pedantic
	        -fstack-protector-all
	        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( agg_sum_test_app PRIVATE "-ldl" )
	
	target_compile_options( group_test_app PRIVATE
	        -Werror
	        -Wall
	        -Wextra
	        -pedantic
	        -fstack-protector-all
	        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( group_test_app PRIVATE "-ldl" )
	
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
	
	add_test( agg_sum_test agg_sum_test_app )
	add_test( group_test group_test_app )
	add_test( join_test join_test_app )
	add_test( project_test project_test_app )
	add_test( select_test select_test_app )
	add_test( operation_lookup_test operation_lookup_test_app )
	add_test( operation_summation_test operation_summation_test_app )
endif()
	