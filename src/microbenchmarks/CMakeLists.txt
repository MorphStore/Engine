if( BUILD_ALL OR BUILD_MICROBMS )
    add_executable( iterator_variants iterator_variants.cpp )
    target_compile_options( iterator_variants PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( iterator_variants PRIVATE "-ldl" )

    #add_executable( agg_sum_benchmark agg_sum_benchmark.cpp )
    #target_compile_options( agg_sum_benchmark PRIVATE
    #                        $<$<CONFIG:DEBUG>:-DDEBUG> )
    #target_link_libraries( agg_sum_benchmark PRIVATE "-ldl" )

    add_executable( select_benchmark select_benchmark.cpp )
    target_compile_options( select_benchmark PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( select_benchmark PRIVATE "-ldl" )

    add_executable( select_uncompr_benchmark select_uncompr.cpp )
    target_compile_options( select_uncompr_benchmark PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( select_uncompr_benchmark PRIVATE "-ldl" )

    add_executable( inc_benchmark inc_benchmark.cpp )
    target_compile_options( inc_benchmark PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( inc_benchmark PRIVATE "-ldl" )

    add_executable( project_benchmark project_benchmark.cpp )
    target_compile_options( project_benchmark PRIVATE
                           # $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( project_benchmark PRIVATE "-ldl" )

    add_executable( calc_binary_benchmark calc_binary_benchmark.cpp )
    target_compile_options( calc_binary_benchmark PRIVATE
                            #$<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( calc_binary_benchmark PRIVATE "-ldl" )

    add_executable( merge_benchmark merge_benchmark.cpp )
    target_compile_options( merge_benchmark PRIVATE
                            -Wno-unused-parameter
                            #$<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( merge_benchmark PRIVATE "-ldl" )

    add_executable( join_benchmark join_benchmark.cpp )
    target_compile_options( join_benchmark PRIVATE
                            -Wno-unused-parameter
                            #$<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( join_benchmark PRIVATE "-ldl" )
endif( BUILD_ALL OR BUILD_MICROBMS )