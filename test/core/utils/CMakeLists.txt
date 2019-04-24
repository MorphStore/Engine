if ( CTEST_ALL OR CTEST_UTILS )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/utils/printing_test_app )
	
	add_executable( printing_test_app printing_test.cpp )
	target_compile_options( printing_test_app PRIVATE
	                        -Werror
	                        -Wall
	                        -Wextra
	                        -pedantic
	                        -fstack-protector-all
	                        $<$<CONFIG:DEBUG>:-DDEBUG> )
	target_link_libraries( printing_test_app PRIVATE "-ldl" )
	add_test( printing_test printing_test_app )
endif()	
