add_executable( operation_lookup_test lookup_test.cpp )
add_executable( operation_summation_test summation_test.cpp )

target_compile_options( operation_lookup_test PRIVATE
        -Werror
        -Wall
        -Wextra
        -pedantic
        -fstack-protector-all
        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( operation_lookup_test PRIVATE "-ldl" )


target_compile_options( operation_summation_test PRIVATE
        -Werror
        -Wall
        -Wextra
        -pedantic
        -fstack-protector-all
        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( operation_summation_test PRIVATE "-ldl" )

