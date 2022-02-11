if( BUILD_ALL OR BUILD_MICROBMS )
    add_executable( iterator_variants iterator_variants.cpp )
    target_compile_options( iterator_variants PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( iterator_variants PRIVATE "-ldl" )

    #add_executable( agg_sum_benchmark agg_sum_benchmark.cpp )
    #target_compile_options( agg_sum_benchmark PRIVATE
    #                        $<$<CONFIG:DEBUG>:-DDEBUG>
    #)
    #target_link_libraries( agg_sum_benchmark PRIVATE "-ldl" )

    add_executable( select_benchmark select_benchmark.cpp )
    target_compile_options( select_benchmark PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( select_benchmark PRIVATE "-ldl" )

    add_executable( select_benchmark_2_t select_benchmark_2.cpp )
    target_compile_options( select_benchmark_2_t PRIVATE
                            -DSELECT_BENCHMARK_2_TIME
                            $<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( select_benchmark_2_t PRIVATE "-ldl" )

    add_executable( select_benchmark_2_d select_benchmark_2.cpp )
    target_compile_options( select_benchmark_2_d PRIVATE
                            -USELECT_BENCHMARK_2_TIME
                            $<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( select_benchmark_2_d PRIVATE "-ldl" )

    add_executable( select_uncompr_benchmark select_uncompr.cpp )
    target_compile_options( select_uncompr_benchmark PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( select_uncompr_benchmark PRIVATE "-ldl" )

    add_executable( inc_benchmark inc_benchmark.cpp )
    target_compile_options( inc_benchmark PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( inc_benchmark PRIVATE "-ldl" )

    add_executable( project_benchmark_t project_benchmark.cpp )
    target_compile_options( project_benchmark_t PRIVATE
                            -DPROJECT_BENCHMARK_TIME
                           # $<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( project_benchmark_t PRIVATE "-ldl" )

    add_executable( project_benchmark_d project_benchmark.cpp )
    target_compile_options( project_benchmark_d PRIVATE
                            -UPROJECT_BENCHMARK_TIME
                           # $<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( project_benchmark_d PRIVATE "-ldl" )

    add_executable( calc_binary_benchmark calc_binary_benchmark.cpp )
    target_compile_options( calc_binary_benchmark PRIVATE
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( calc_binary_benchmark PRIVATE "-ldl" )

    add_executable( merge_benchmark merge_benchmark.cpp )
    target_compile_options( merge_benchmark PRIVATE
                            -Wno-unused-parameter
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( merge_benchmark PRIVATE "-ldl" )

    add_executable( join_benchmark join_benchmark.cpp )
    target_compile_options( join_benchmark PRIVATE
                            -Wno-unused-parameter
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( join_benchmark PRIVATE "-ldl" )

    add_executable( semi_join_benchmark semi_join_benchmark.cpp )
    target_compile_options( semi_join_benchmark PRIVATE
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( semi_join_benchmark PRIVATE "-ldl" )

    add_executable( select_sum_benchmark select_sum_benchmark.cpp )
    target_compile_options( select_sum_benchmark PRIVATE
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( select_sum_benchmark PRIVATE "-ldl" )

    add_executable( otf_morphing_example_1 otf_morphing_example_1.cpp )
    target_compile_options( otf_morphing_example_1 PRIVATE
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( otf_morphing_example_1 PRIVATE "-ldl" )

    # *************************************************************************
    # compression_data_benchmark
    # *************************************************************************
    # The following 6 targets represent variants of the
    # compression_data_benchmark. The variants result from the combinations of
    # two macro constants:
    # - COMPRESSION_DATA_BENCHMARK_DATA_SOURCE: 3 data sources (hist, distr,
    #   datafile)
    # - COMPRESSION_DATA_BENCHMARK_RUN: either executing the compression
    #   algorithms (D, defined), or only analyzing the data characteristics (U,
    #   undefined)
    # See compression_data_benchmark.cpp for how these macro constants are used.

    add_executable( compression_data_benchmark_hist_run compression_data_benchmark.cpp )
    target_compile_options( compression_data_benchmark_hist_run PRIVATE
                            -DCOMPRESSION_DATA_BENCHMARK_DATA_SOURCE=1
                            -DCOMPRESSION_DATA_BENCHMARK_RUN
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( compression_data_benchmark_hist_run PRIVATE "-ldl" )

    add_executable( compression_data_benchmark_hist_datach compression_data_benchmark.cpp )
    target_compile_options( compression_data_benchmark_hist_datach PRIVATE
                            -DCOMPRESSION_DATA_BENCHMARK_DATA_SOURCE=1
                            -UCOMPRESSION_DATA_BENCHMARK_RUN
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( compression_data_benchmark_hist_datach PRIVATE "-ldl" )

    add_executable( compression_data_benchmark_distr_run compression_data_benchmark.cpp )
    target_compile_options( compression_data_benchmark_distr_run PRIVATE
                            -DCOMPRESSION_DATA_BENCHMARK_DATA_SOURCE=2
                            -DCOMPRESSION_DATA_BENCHMARK_RUN
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( compression_data_benchmark_distr_run PRIVATE "-ldl" )

    add_executable( compression_data_benchmark_distr_datach compression_data_benchmark.cpp )
    target_compile_options( compression_data_benchmark_distr_datach PRIVATE
                            -DCOMPRESSION_DATA_BENCHMARK_DATA_SOURCE=2
                            -UCOMPRESSION_DATA_BENCHMARK_RUN
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( compression_data_benchmark_distr_datach PRIVATE "-ldl" )

    add_executable( compression_data_benchmark_datafile_run compression_data_benchmark.cpp )
    target_compile_options( compression_data_benchmark_datafile_run PRIVATE
                            -DCOMPRESSION_DATA_BENCHMARK_DATA_SOURCE=3
                            -DCOMPRESSION_DATA_BENCHMARK_RUN
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( compression_data_benchmark_datafile_run PRIVATE "-ldl" )

    add_executable( compression_data_benchmark_datafile_datach compression_data_benchmark.cpp )
    target_compile_options( compression_data_benchmark_datafile_datach PRIVATE
                            -DCOMPRESSION_DATA_BENCHMARK_DATA_SOURCE=3
                            -UCOMPRESSION_DATA_BENCHMARK_RUN
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( compression_data_benchmark_datafile_datach PRIVATE "-ldl" )
endif( BUILD_ALL OR BUILD_MICROBMS )
