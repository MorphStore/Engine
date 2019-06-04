if ( CTEST_ALL OR CTEST_MEMORY )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/memory_test_app )
	add_executable( memory_test_app memory_test.cpp )
	target_compile_options( memory_test_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( memory_test_app PRIVATE "-ldl" )

	add_test( memory_test memory_test_app )
endif()