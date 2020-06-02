if( BUILD_ALL OR BUILD_EXAMPLES )
    #add_executable( basic basic.cpp )
    #link_directories (..)
    #target_compile_options( basic PRIVATE
    #                       # $<$<CONFIG:DEBUG>:-DDEBUG> 
    #                       )
    #target_link_libraries( basic PRIVATE "-ldl" "-lpthread" "-lnuma" "-lrt" "-lPCM")

    add_executable( frame frame.cpp )
    link_directories (..)
    target_compile_options( frame PRIVATE
                           # $<$<CONFIG:DEBUG>:-DDEBUG> 
                       )
    target_link_libraries( frame PRIVATE "-ldl" "-lpthread" "-lnuma" "-lrt" "-lPCM")

    #add_executable( frame0 frame0.cpp )
    #link_directories (..)
    #target_compile_options( frame0 PRIVATE
    #                       # $<$<CONFIG:DEBUG>:-DDEBUG> 
    #                   )
    #target_link_libraries( frame0 PRIVATE "-ldl" "-lpthread" "-lnuma" "-lrt" "-lPCM")

    #add_executable( frame_ssb frame_ssb.cpp )
    #link_directories (..)
    #target_compile_options( frame_ssb PRIVATE
    #                       # $<$<CONFIG:DEBUG>:-DDEBUG> 
    #                   )
    #target_link_libraries( frame_ssb PRIVATE "-ldl" "-lpthread" "-lnuma" "-lrt" "-lPCM")

    add_executable( select_sum_query select_sum_query.cpp )
    target_compile_options( select_sum_query PRIVATE
                           # $<$<CONFIG:DEBUG>:-DDEBUG> 
                           )
    target_link_libraries( select_sum_query PRIVATE "-ldl" "-lpthread" "-lnuma" )

    add_executable( example_query example_query.cpp )
    target_compile_options( example_query PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( example_query PRIVATE "-ldl" "-lpthread" "-lnuma" )

    add_executable( variant_executor_usage variant_executor_usage.cpp )
    target_compile_options( variant_executor_usage PRIVATE
                            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( variant_executor_usage PRIVATE "-ldl" "-lpthread" "-lnuma" )
endif( BUILD_ALL OR BUILD_EXAMPLES )