if ( CTEST_ALL OR CTEST_VECTOR )
    add_executable( vector_test_app main.cpp )

    if ( CAVX512 )
    target_compile_options( vector_test_app PRIVATE
#            -Werror
#            -Wall
#            -Wextra
#            -pedantic
            -fstack-protector-all
            -msse4.2 -mavx2 -msse2
           # -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl 
           
            $<$<CONFIG:DEBUG>:-DDEBUG> )
    else()
    target_compile_options( vector_test_app PRIVATE
#            -Werror
#            -Wall
#            -Wextra
#            -pedantic
            -fstack-protector-all
            -msse4.2 -mavx2 -msse2
           
            $<$<CONFIG:DEBUG>:-DDEBUG> )
    endif()

    target_link_libraries( vector_test_app PRIVATE "-ldl" )
    
    add_test( vector_test vector_test_app )
endif()