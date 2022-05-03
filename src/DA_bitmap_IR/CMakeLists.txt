if ( BUILD_ALL OR BUILD_DA_MICROBMS)

    # ---------------- MICRO BENCHMARK 1 -------------------

    add_executable( micro_benchmark_1_uniform_scalar micro_benchmark_1_uniform_scalar.cpp )
    target_compile_options( micro_benchmark_1_uniform_scalar PRIVATE
            $<$<CONFIG:DEBUG>:-DDEBUG>
            )
    target_link_libraries( micro_benchmark_1_uniform_scalar PRIVATE "-ldl" )

    add_executable( micro_benchmark_1_uniform_avx2 micro_benchmark_1_uniform_avx2.cpp )
    target_compile_options( micro_benchmark_1_uniform_avx2 PRIVATE
            $<$<CONFIG:DEBUG>:-DDEBUG>
            )
    target_link_libraries( micro_benchmark_1_uniform_avx2 PRIVATE "-ldl" )

    add_executable( micro_benchmark_1_uniform_avx512 micro_benchmark_1_uniform_avx512.cpp )
    target_compile_options( micro_benchmark_1_uniform_avx512 PRIVATE
            $<$<CONFIG:DEBUG>:-DDEBUG>
            )
    target_link_libraries( micro_benchmark_1_uniform_avx512 PRIVATE "-ldl" )

    # ---------------- MICRO BENCHMARK 2 -------------------

    add_executable( micro_benchmark_2_uniform_scalar micro_benchmark_2_uniform_scalar.cpp)
    target_compile_options( micro_benchmark_2_uniform_scalar PRIVATE
            $<$<CONFIG:DEBUG>:-DDEBUG>
            )
    target_link_libraries( micro_benchmark_2_uniform_scalar PRIVATE "-ldl" )

    # ---------------- MICRO BENCHMARK 3 -------------------

    add_executable( micro_benchmark_3_uniform_sorted_scalar micro_benchmark_3_uniform_sorted_unique_scalar.cpp)
    target_compile_options( micro_benchmark_3_uniform_sorted_scalar PRIVATE
            $<$<CONFIG:DEBUG>:-DDEBUG>
            )
    target_link_libraries( micro_benchmark_3_uniform_sorted_scalar PRIVATE "-ldl" )

    add_executable( micro_benchmark_3_uniform_sorted_vectorized micro_benchmark_3_uniform_sorted_unique_vectorized.cpp)
    target_compile_options( micro_benchmark_3_uniform_sorted_vectorized PRIVATE
            $<$<CONFIG:DEBUG>:-DDEBUG>
            )
    target_link_libraries( micro_benchmark_3_uniform_sorted_vectorized PRIVATE "-ldl" )

    # ---------------- MICRO BENCHMARK 4 -------------------

    add_executable( micro_benchmark_4_uniform_sorted_unique_scalar micro_benchmark_4_uniform_sorted_unique_scalar.cpp)
    target_compile_options( micro_benchmark_4_uniform_sorted_unique_scalar PRIVATE
            $<$<CONFIG:DEBUG>:-DDEBUG>
            )
    target_link_libraries( micro_benchmark_4_uniform_sorted_unique_scalar PRIVATE "-ldl" )

endif()