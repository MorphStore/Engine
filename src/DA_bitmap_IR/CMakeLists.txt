if ( BUILD_ALL OR BUILD_DA_MICROBMS)
    #add_executable( micro_benchmark_1_asc_scalar micro_benchmark_1_asc_scalar.cpp )
    #target_compile_options( micro_benchmark_1_asc_scalar PRIVATE
    #        $<$<CONFIG:DEBUG>:-DDEBUG>
    #        )
    #target_link_libraries( micro_benchmark_1_asc_scalar PRIVATE "-ldl" )

    add_executable( micro_benchmark_1_uniform_scalar micro_benchmark_1_uniform_scalar.cpp )
    target_compile_options( micro_benchmark_1_uniform_scalar PRIVATE
            $<$<CONFIG:DEBUG>:-DDEBUG>
            )
    target_link_libraries( micro_benchmark_1_uniform_scalar PRIVATE "-ldl" )
endif()