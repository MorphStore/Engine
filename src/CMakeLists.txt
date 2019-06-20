add_subdirectory( examples )
add_subdirectory( microbenchmarks )

# There might be automatically generated subdirectories for the Star Schema
# Benchmark (SSB), possibly with different scale factors. The following lines
# add all of them.
file( GLOB SSBSubdirs ssb_sf* )
foreach( SSBSubdir ${SSBSubdirs} )
    if( IS_DIRECTORY ${SSBSubdir} )
        add_subdirectory( ${SSBSubdir} )
    endif()
endforeach()