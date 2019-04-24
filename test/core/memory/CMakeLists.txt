if ( CTEST_ALL OR CTEST_MEMORY )
	message( "Core/Memory" )
	add_executable( memory_test_app memory_test.cpp )
	target_compile_options( memory_test_app PRIVATE
	                        -Werror
	                        -Wall
	                        -Wextra
	                        -pedantic
	                        -fstack-protector-all
	                        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( memory_test_app PRIVATE "-ldl" )

	add_executable( mmap_test_app mmap_test.cpp )
	target_compile_options( mmap_test_app PRIVATE
                            -std=c++17
	                        -Werror
	                        -Wall
	                        -Wextra
	                        -pedantic
	                        -fstack-protector-all
	                        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( mmap_test_app PRIVATE "-ldl" )

	add_test( memory_test memory_test_app )
	add_test( mmap_test mmap_test_app )
endif()
