add_executable( agg_sum_test agg_sum_test.cpp )
add_executable( group_test group_test.cpp )
add_executable( join_test join_test.cpp )
add_executable( project_test project_test.cpp )
add_executable( select_test select_test.cpp )
add_executable( operation_lookup_test lookup_test.cpp )
add_executable( operation_summation_test summation_test.cpp )

target_compile_options( agg_sum_test PRIVATE
        -Werror
        -Wall
        -Wextra
        -pedantic
        -fstack-protector-all
        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( agg_sum_test PRIVATE "-ldl" )

target_compile_options( group_test PRIVATE
        -Werror
        -Wall
        -Wextra
        -pedantic
        -fstack-protector-all
        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( group_test PRIVATE "-ldl" )

target_compile_options( join_test PRIVATE
        -Werror
        -Wall
        -Wextra
        -pedantic
        -fstack-protector-all
        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( join_test PRIVATE "-ldl" )

target_compile_options( project_test PRIVATE
        -Werror
        -Wall
        -Wextra
        -pedantic
        -fstack-protector-all
        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( project_test PRIVATE "-ldl" )

target_compile_options( select_test PRIVATE
        -Werror
        -Wall
        -Wextra
        -pedantic
        -fstack-protector-all
        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( select_test PRIVATE "-ldl" )

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

