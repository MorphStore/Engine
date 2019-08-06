if ( CTEST_ALL OR CTEST_QUERIES )
add_executable( select_sum_query select_sum_query.cpp )
target_compile_options( select_sum_query PRIVATE
                       # -Werror
                       # -Wall
                       # -Wextra
                       # -pedantic
                       # -fstack-protector-all
                       # $<$<CONFIG:DEBUG>:-DDEBUG> 
                       )
target_link_libraries( select_sum_query PRIVATE "-ldl" )

add_executable( example_query example_query.cpp )
target_compile_options( example_query PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( example_query PRIVATE "-ldl" )

add_test( example_query_test example_query )
add_test( select_sum_query_test select_sum_query )

endif(CTEST_ALL OR CTEST_QUERIES)

if ( CTEST_ALL )
add_executable( variant_executor_usage variant_executor_usage.cpp )
target_compile_options( variant_executor_usage PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( variant_executor_usage PRIVATE "-ldl" )
endif ( CTEST_ALL )