if( BUILD_ALL OR BUILD_PARALLELIZATION )
#    morph_flag(-DDEBUG)
    add_executable( virtual_vector_query virtual_vector_query.cpp )
    target_link_libraries( virtual_vector_query PRIVATE "-ldl" )

    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -pthread -Og")

    message("Compiler Flags: ${CMAKE_CXX_FLAGS}")


    target_sources(virtual_vector_query
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include/core/utils/monitoring/Monitor.h
        ${PROJECT_SOURCE_DIR}/include/core/utils/monitoring/MonitorTask.h
    )


endif( BUILD_ALL OR BUILD_PARALLELIZATION )
