add_executable( select_sum_query select_sum_query.cpp )
target_compile_options( select_sum_query PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
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

