if ( CTEST_ALL OR CTEST_MORPHING )
	add_executable( morphing_test_app morphing_test.cpp )
	target_compile_options( morphing_test_app PRIVATE
	                        -Werror
	                        -Wall
	                        -Wextra
	                        -pedantic
	                        -fstack-protector-all
	                        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( morphing_test_app PRIVATE "-ldl" )
	add_test( morphing_test morphing_test_app )
endif()