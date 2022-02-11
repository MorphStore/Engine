if( BUILD_ALL OR BUILD_EXAMPLES )
    add_executable( select_sum_query select_sum_query.cpp)
    target_compile_options( select_sum_query PRIVATE
                           # $<$<CONFIG:DEBUG>:-DDEBUG>
                           )
    target_link_libraries( select_sum_query PRIVATE "-ldl" )

    add_executable( example_query example_query.cpp)
    target_compile_options( example_query PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( example_query PRIVATE "-ldl" )

#    add_executable( gpu_test gpu_test.cu )
#    target_compile_options( gpu_test PRIVATE
#                            $<$<CONFIG:DEBUG>:-DDEBUG> )
#    target_link_libraries( gpu_test PRIVATE "-ldl" )
#    set_property(TARGET gpu_test PROPERTY CUDA_ARCHITECTURES 75)

#    add_executable( gpu_operator_test gpu_operator_test.cu )
#    target_compile_options( gpu_operator_test PRIVATE
#                            $<$<CONFIG:DEBUG>:-DDEBUG> )
#    target_link_libraries( gpu_operator_test PRIVATE "-ldl" )
#    set_property(TARGET gpu_operator_test PROPERTY CUDA_ARCHITECTURES 75)

    add_executable( variant_executor_usage variant_executor_usage.cpp)
    target_compile_options( variant_executor_usage PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( variant_executor_usage PRIVATE "-ldl" )

    add_executable( group_by_test group_by_test.cpp)
    target_compile_options( group_by_test PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( group_by_test PRIVATE "-ldl" )
endif( BUILD_ALL OR BUILD_EXAMPLES )
