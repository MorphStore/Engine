if ( CTEST_ALL OR CTEST_VECTOR )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/vector/vector_test_app )
	add_executable( vector_test_app main.cpp )

    target_compile_options( vector_test_app PRIVATE
#            -Werror
#            -Wall
#            -Wextra
#            -pedantic
            -fstack-protector-all
            -msse4.2 -msse2
           # -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl 
           
            $<$<CONFIG:DEBUG>:-DDEBUG> )
  
    target_link_libraries( vector_test_app PRIVATE "-ldl" )
    
    add_test( vector_test vector_test_app )
endif()