if ( CTEST_ALL OR CTEST_VECTOR )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/vector_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/agg_sum_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/hash_set_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/equi_join_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/semi_join_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/calc_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/project_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/select_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/intersect_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/merge_app )
#	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/my_operator_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/group_unary_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/group_binary_app )

#	add_executable( vector_test_app main.cpp )
	add_executable( agg_sum_app operators/agg_sum_whole_column_test.cpp )
#	add_executable( hash_set_app operators/hash_set_test.cpp )
	add_executable( equi_join_app operators/equi_join_test.cpp )
	add_executable( semi_join_app operators/semi_join_test.cpp )
	add_executable( calc_app operators/calc_test.cpp )
	add_executable( project_app operators/project_test.cpp )
	add_executable( select_app operators/select_test.cpp )
	add_executable( intersect_app operators/intersect_test.cpp )
	add_executable( merge_app operators/merge_test.cpp )
#	add_executable( my_operator_app operators/my_operator.cpp )
	add_executable( group_unary_app operators/group_unary_test.cpp )
	add_executable( group_binary_app operators/group_binary_test.cpp )

#	target_compile_options( vector_test_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#		-msse4.2 -msse2
#	)
#	target_link_libraries( vector_test_app PRIVATE "-ldl" )

	target_compile_options( agg_sum_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( agg_sum_app PRIVATE "-ldl" )

#	target_compile_options( hash_set_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( hash_set_app PRIVATE "-ldl" )

	target_compile_options( equi_join_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( equi_join_app PRIVATE "-ldl" )

	target_compile_options( semi_join_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( semi_join_app PRIVATE "-ldl" )

	target_compile_options( calc_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( calc_app PRIVATE "-ldl" )

	target_compile_options( project_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( project_app PRIVATE "-ldl" )

	target_compile_options( select_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( select_app PRIVATE "-ldl" )

	target_compile_options( intersect_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( intersect_app PRIVATE "-ldl" )

	target_compile_options( merge_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( merge_app PRIVATE "-ldl" )

#	target_compile_options( my_operator_app PRIVATE
#		# space to add custom flags for THIS SPECIFIC TARGET
#	)
#	target_link_libraries( my_operator_app PRIVATE "-ldl" )

	target_compile_options( group_unary_app PRIVATE
			# space to add custom flags for THIS SPECIFIC TARGET
			)
	target_link_libraries( group_unary_app PRIVATE "-ldl" )

	target_compile_options( group_binary_app PRIVATE
			# space to add custom flags for THIS SPECIFIC TARGET
			)
	target_link_libraries( group_binary_app PRIVATE "-ldl" )

#	add_test( vector_test vector_test_app )
	add_test( agg_sum_vectorized_test agg_sum_app )
#	add_test( hash_set_vectorized_test hash_set_app )
	add_test( equi_join_vectorized_test equi_join_app )
	add_test( semi_join_vectorized_test semi_join_app )
	add_test( calc_test calc_app )
	add_test( project_test project_app )
	add_test( select_test select_app )
	add_test( intersect_test intersect_app )
	add_test( merge_test merge_app )
	add_test( my_operator_test my_operator_app )
	add_test( group_unary_test group_unary_app )
	add_test( group_binary_test group_binary_app )
endif()