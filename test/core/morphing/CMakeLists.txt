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

if ( CTEST_ALL OR CTEST_MORPHING )
	add_executable( dynamic_vbp_test_app dynamic_vbp_test.cpp )
	target_compile_options( dynamic_vbp_test_app PRIVATE
	                        -Werror
	                        -Wall
	                        -Wextra
	                        -pedantic
	                        -fstack-protector-all
	                        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( dynamic_vbp_test_app PRIVATE "-ldl" )
	add_test( dynamic_vbp_test dynamic_vbp_test_app )
endif()