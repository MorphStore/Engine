if( BUILD_ALL OR BUILD_PARALLELIZATION )
    add_executable( virtual_vector_query virtual_vector_query.cpp)
    target_link_libraries( virtual_vector_query PRIVATE "-ldl" )

    target_sources(virtual_vector_query
        PUBLIC
            ../../../include/core/utils/monitoring/Monitor.h
            ../../../include/core/utils/monitoring/MonitorTask.h
    )

    add_executable( virtual_vector_template_test virtual_vector_template_test.cpp)
    target_link_libraries( virtual_vector_template_test PRIVATE "-ldl" )

    add_executable( meta_operator_prototype_test meta_operator_prototype_test.cpp)
    target_link_libraries( meta_operator_prototype_test PRIVATE "-ldl" )

    add_executable( overhead_benchmark_meta_operator overhead_benchmark_meta_operator.cpp)
    target_link_libraries( overhead_benchmark_meta_operator PUBLIC -ldl -lpthread )

else( BUILD_ALL OR BUILD_PARALLELIZATION )
    message(">> Set CMake variable BUILD_PARALLELIZATION to build parallelization targets")
endif( BUILD_ALL OR BUILD_PARALLELIZATION )
