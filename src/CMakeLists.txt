add_subdirectory( calibration )
add_subdirectory( examples )
add_subdirectory( microbenchmarks )
add_subdirectory( DA_bitmap_IR ) # subdirectory for micro benchmarks used in the diploma thesis

# There might be automatically generated subdirectories for the Star Schema
# Benchmark (SSB), possibly with different scale factors. The following lines
# add all of them.
file( GLOB SSBSubdirs ssb_sf* )
foreach( SSBSubdir ${SSBSubdirs} )
    if( IS_DIRECTORY ${SSBSubdir} )
        add_subdirectory( ${SSBSubdir} )
    endif()
endforeach()