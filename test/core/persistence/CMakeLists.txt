if ( CTEST_ALL OR CTEST_PERSISTENCE )
	add_executable( binary_io_test_app binary_io_test.cpp )
	target_compile_options( binary_io_test_app PRIVATE
	                        -Werror
	                        -Wall
	                        -Wextra
	                        -pedantic
	                        -fstack-protector-all
	                        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( binary_io_test_app PRIVATE "-ldl" )
	add_test( binary_io_test binary_io_test_app )
endif()
