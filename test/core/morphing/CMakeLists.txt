if ( CTEST_ALL OR CTEST_MORPHING )
	add_executable( static_vbp_test_app static_vbp_test.cpp )
	target_compile_options( static_vbp_test_app PRIVATE
	                        -Werror
	                        -Wall
	                        -Wextra
	                        -pedantic
	                        -fstack-protector-all
	                        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( static_vbp_test_app PRIVATE "-ldl" )
	add_test( static_vbp_test static_vbp_test_app )
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