add_executable( scalable_agg_sum scalable_agg_sum.cpp )
target_compile_options( scalable_agg_sum PRIVATE
                   # $<$<CONFIG:DEBUG>:-DDEBUG> 
                   )
target_link_libraries( scalable_agg_sum PRIVATE "-ldl" )