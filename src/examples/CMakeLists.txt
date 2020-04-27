if( BUILD_ALL OR BUILD_EXAMPLES )
    add_executable( select_sum_query select_sum_query.cpp )
    target_compile_options( select_sum_query PRIVATE
                           # $<$<CONFIG:DEBUG>:-DDEBUG> 
                           )
    target_link_libraries( select_sum_query PRIVATE "-ldl" )

    add_executable( example_query example_query.cpp )
    target_compile_options( example_query PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( example_query PRIVATE "-ldl" )

    add_executable( variant_executor_usage variant_executor_usage.cpp )
    target_compile_options( variant_executor_usage PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( variant_executor_usage PRIVATE "-ldl" )

    add_executable( test_type_packing test_type_packing.cpp )
    target_compile_options( test_type_packing PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( test_type_packing PRIVATE "-ldl" )
endif( BUILD_ALL OR BUILD_EXAMPLES )