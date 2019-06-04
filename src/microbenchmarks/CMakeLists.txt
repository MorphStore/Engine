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