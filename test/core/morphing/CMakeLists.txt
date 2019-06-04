if ( CTEST_ALL OR CTEST_MORPHING )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/static_vbp_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/dynamic_vbp_test_app )
	
	add_executable( static_vbp_test_app static_vbp_test.cpp )
	target_compile_options( static_vbp_test_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( static_vbp_test_app PRIVATE "-ldl" )
	add_test( static_vbp_test static_vbp_test_app )

	add_executable( dynamic_vbp_test_app dynamic_vbp_test.cpp )
	target_compile_options( dynamic_vbp_test_app PRIVATE
		# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_link_libraries( dynamic_vbp_test_app PRIVATE "-ldl" )
	add_test( dynamic_vbp_test dynamic_vbp_test_app )
endif()