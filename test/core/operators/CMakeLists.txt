if ( CTEST_ALL OR CTEST_OPERATORS )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/select_nec_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/agg_sum_group_based_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/agg_sum_whole_column_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/calc_binary_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/calc_unary_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/group_binary_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/group_unary_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/intersect_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/join_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/left_semi_nto1_nested_loop_join_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/merge_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/project_test_app )
#
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/select_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/operation_summation_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/random_access_test_app )
#
#IF(CAVXTWO OR CAVX512)
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/project_benchmark_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/operations_vector_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/group_unary_vector_test_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/operators/group_binary_vector_test_app )
#ENDIF(CAVXTWO  OR CAVX512)

	add_executable( select_nec_app select_nec_test.cpp )
#	add_executable( agg_sum_group_based_test_app agg_sum_group_based_test.cpp )
#	add_executable( agg_sum_whole_column_test_app agg_sum_whole_column_test.cpp )
#	add_executable( calc_binary_test_app calc_binary_test.cpp )
#	add_executable( calc_unary_test_app calc_unary_test.cpp )
#	add_executable( group_binary_test_app group_binary_test.cpp )
#	add_executable( group_unary_test_app group_unary_test.cpp )
#	add_executable( intersect_test_app intersect_test.cpp )
#	add_executable( join_test_app join_test.cpp )
#	add_executable( left_semi_nto1_nested_loop_join_test_app left_semi_nto1_nested_loop_join_test.cpp )
#	add_executable( merge_test_app merge_test.cpp )
#	add_executable( project_test_app project_test.cpp )
#	add_executable( select_test_app select_test.cpp )
#	add_executable( operation_summation_test_app summation_test.cpp )
#	add_executable( random_access_test_app random_access_test.cpp )

#IF(CAVXTWO  OR CAVX512)
#	add_executable( project_benchmark_test_app project_benchmark_test.cpp )
#	add_executable( operations_vector_test_app vector_test.cpp )
#	add_executable( group_unary_vector_test_app group_unary_vectorized_test.cpp)
#	add_executable( group_binary_vector_test_app group_binary_vectorized_test.cpp)
#ENDIF(CAVXTWO  OR CAVX512)

#	target_compile_options( select_nec_app PRIVATE
#			# space to add custom flags for THIS SPECIFIC TARGET
#			)
#	target_link_libraries( select_nec_app PRIVATE  )
	
#	target_compile_options( agg_sum_group_based_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( agg_sum_group_based_test_app PRIVATE "-ldl" )
#
#	target_compile_options( agg_sum_whole_column_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( agg_sum_whole_column_test_app PRIVATE "-ldl" )
#
#	target_compile_options( calc_binary_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( calc_binary_test_app PRIVATE "-ldl" )
#
#	target_compile_options( calc_unary_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( calc_unary_test_app PRIVATE "-ldl" )
#
#	target_compile_options( group_binary_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( group_binary_test_app PRIVATE "-ldl" )
#
#	target_compile_options( group_unary_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( group_unary_test_app PRIVATE "-ldl" )
#
#	target_compile_options( intersect_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( intersect_test_app PRIVATE "-ldl" )
#
#	target_compile_options( join_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( join_test_app PRIVATE "-ldl" )
#
#	target_compile_options( left_semi_nto1_nested_loop_join_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( left_semi_nto1_nested_loop_join_test_app PRIVATE "-ldl" )
#
#	target_compile_options( merge_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( merge_test_app PRIVATE "-ldl" )
#
#	target_compile_options( project_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( project_test_app PRIVATE "-ldl" )
#
#	target_compile_options( select_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( select_test_app PRIVATE "-ldl" )
#
#	target_compile_options( operation_summation_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( operation_summation_test_app PRIVATE "-ldl" )
#
#	target_compile_options( random_access_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( random_access_test_app PRIVATE "-ldl" )
#
#IF(CAVXTWO  OR CAVX512)
#	target_compile_options( project_benchmark_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#		-msse4.2
#		-mavx2
#	)
#	target_link_libraries( project_benchmark_test_app PRIVATE "-ldl" )
#
#	target_compile_options( operations_vector_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#		-msse4.2
#		-mavx2
#	)
#	target_link_libraries( operations_vector_test_app PRIVATE "-ldl" )
#
#	target_compile_options( group_unary_vector_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#		-msse4.2
#		-mavx2
#	)
#	target_link_libraries( group_unary_vector_test_app PRIVATE "-ldl" )
#
#	target_compile_options( group_binary_vector_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#		-msse4.2
#		-mavx2
#	)
#	target_link_libraries( group_binary_vector_test_app PRIVATE "-ldl" )
#ENDIF(CAVXTWO  OR CAVX512)

	add_test( select_nec_app_test select_nec_app )
#	add_test( agg_sum_group_based_test agg_sum_group_based_test_app )
#	add_test( agg_sum_whole_column_test agg_sum_whole_column_test_app )
#	add_test( calc_binary_test calc_binary_test_app )
#	add_test( calc_unary_test calc_unary_test_app )
#	add_test( group_binary_test group_binary_test_app )
#	add_test( group_unary_test group_unary_test_app )
#	add_test( intersect_test intersect_test_app )
#	add_test( join_test join_test_app )
#	add_test( left_semi_nto1_nested_loop_join_test join_test_app )
#	add_test( merge_test merge_test_app )
#	add_test( project_test project_test_app )
#	add_test( select_test select_test_app )
#	add_test( operation_summation_test operation_summation_test_app )
#	add_test( random_access_test random_access_test_app )
#
#IF(CAVXTWO  OR CAVX512)
#	add_test( project_benchmark_test project_benchmark_test_app )
#	add_test( operations_vector_test operations_vector_test_app )
#	add_test( group_unary_vector_test group_unary_vector_test_app )
#	add_test( group_binary_vector_test group_binary_vector_test_app )
#ENDIF(CAVXTWO  OR CAVX512)

endif( CTEST_ALL OR CTEST_OPERATORS)
	