if( BUILD_ALL OR BUILD_PARALLELIZATION )
#    morph_flag(-DDEBUG)
    add_executable( const_column_test const_column_test.cpp)
    target_link_libraries( const_column_test PRIVATE "-ldl" )


endif( BUILD_ALL OR BUILD_PARALLELIZATION )
