add_executable( iterator_variants iterator_variants.cpp )
target_compile_options( iterator_variants PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( iterator_variants PRIVATE "-ldl" )

add_executable( agg_sum_benchmark agg_sum_benchmark.cpp )
target_compile_options( agg_sum_benchmark PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( agg_sum_benchmark PRIVATE "-ldl" )

add_executable( select_benchmark select_benchmark.cpp )
target_compile_options( select_benchmark PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( select_benchmark PRIVATE "-ldl" )

add_executable( inc_benchmark inc_benchmark.cpp )
target_compile_options( inc_benchmark PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( inc_benchmark PRIVATE "-ldl" )