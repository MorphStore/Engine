add_executable( select_sum_query select_sum_query.cpp )
target_compile_options( select_sum_query PRIVATE
                   # $<$<CONFIG:DEBUG>:-DDEBUG> 
                   )
target_link_libraries( select_sum_query PRIVATE "-ldl" )

