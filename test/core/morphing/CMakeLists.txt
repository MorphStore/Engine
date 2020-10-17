if ( CTEST_ALL OR CTEST_MORPHING )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/delta_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/group_simple_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/k_wise_ns_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/vbp_test_app )
	
	add_executable( delta_test_app delta_test.cpp )
	add_executable( group_simple_test_app group_simple_test.cpp )
	add_executable( k_wise_ns_test_app k_wise_ns_test.cpp )
	add_executable( vbp_test_app vbp_test.cpp )

	target_compile_options( delta_test_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_compile_options( group_simple_test_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_compile_options( k_wise_ns_test_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_compile_options( vbp_test_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)

	target_link_libraries( delta_test_app PRIVATE "-ldl" )
	target_link_libraries( group_simple_test_app PRIVATE "-ldl" )
	target_link_libraries( k_wise_ns_test_app PRIVATE "-ldl" )
	target_link_libraries( vbp_test_app PRIVATE "-ldl" )

	add_test( delta_test delta_test_app )
	add_test( group_simple_test group_simple_test_app )
	add_test( k_wise_ns_test k_wise_ns_test_app )
	add_test( vbp_test vbp_test_app )
endif()