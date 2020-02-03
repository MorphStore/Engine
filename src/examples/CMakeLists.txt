add_executable( agg_sum_example agg_sum_example.cpp )
target_compile_options( agg_sum_example PRIVATE
                   # $<$<CONFIG:DEBUG>:-DDEBUG> 
                   )
target_link_libraries( agg_sum_example PRIVATE "-ldl" )

