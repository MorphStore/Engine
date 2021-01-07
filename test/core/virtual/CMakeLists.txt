if ( CTEST_ALL OR CTEST_VIRTUAL )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/utils/threading_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/utils/partitioning_test_app )

	## threading test
	add_executable( test_threading_app threading_test.cpp)
	target_compile_options( test_threading_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( test_threading_app PRIVATE "-ldl" )
	add_test( threading test_threading_app )

	## partitioning test
	add_executable( test_partitioning_app partitioning_test.cpp )
	target_compile_options( test_partitioning_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( test_partitioning_app PRIVATE "-ldl" )
	add_test( partitioning test_partitioning_app )
endif()
