#add_executable( scalable_agg_sum scalable_agg_sum.cpp )
#target_compile_options( scalable_agg_sum PRIVATE
                   # $<$<CONFIG:DEBUG>:-DDEBUG> 
#                   )
#target_link_libraries( scalable_agg_sum PRIVATE "-ldl" )
add_executable( scalable_project scalable_project.cpp )
target_compile_options( scalable_project PRIVATE
                   # $<$<CONFIG:DEBUG>:-DDEBUG> 
                  )
target_link_libraries( scalable_project PRIVATE "-ldl" )
#add_executable( benchmark_project benchmark_project.cpp )
#target_compile_options( benchmark_project PRIVATE
                   # $<$<CONFIG:DEBUG>:-DDEBUG> 
#                   )
#target_link_libraries( benchmark_project PRIVATE "-ldl" )
add_executable( benchmarks benchmarks.cpp )
target_compile_options( benchmarks PRIVATE
                   # $<$<CONFIG:DEBUG>:-DDEBUG> 
                  )
target_link_libraries( benchmarks PRIVATE "-ldl" )