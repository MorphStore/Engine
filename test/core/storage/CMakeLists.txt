if ( CTEST_ALL OR CTEST_STORAGE )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/storage_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/column_gen_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/PartitionedColumn_unittest_app )


	add_executable( storage_test_app storage_test.cpp )
	target_compile_options( storage_test_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( storage_test_app PRIVATE "-ldl" )
	add_test( storage_test storage_test_app )


	add_executable( column_gen_test_app column_gen_test.cpp )
	target_compile_options( storage_test_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( column_gen_test_app PRIVATE "-ldl" )
	add_test( column_gen_test column_gen_test_app )


	add_executable( PartitionedColumn_unittest_app PartitionedColumn_unittest.cpp )
	target_compile_options( PartitionedColumn_unittest_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( PartitionedColumn_unittest_app PRIVATE "-ldl" )
	add_test( PartitionedColumn_unittest PartitionedColumn_unittest_app )

endif()
