if( BUILD_ALL OR BUILD_CALIB )
    add_executable( bw_prof bw_prof.cpp )
    target_compile_options( bw_prof PRIVATE
                            #$<$<CONFIG:DEBUG>:-DDEBUG>
    )
    target_link_libraries( bw_prof PRIVATE "-ldl" )
endif( BUILD_ALL OR BUILD_CALIB )