cmake_minimum_required( VERSION 3.10 )
project( MorphStore )

set( CMAKE_CXX_STANDARD 14 )

# Get the current working branch
execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)


add_definitions("-DGIT_COMMIT_HASH=${GIT_COMMIT_HASH}")
add_definitions("-DGIT_BRANCH=${GIT_BRANCH}")

configure_file(
   ${CMAKE_SOURCE_DIR}/include/core/utils/cmake_template.h.in
   ${CMAKE_BINARY_DIR}/generated/cmake_template.h
)

include_directories(${CMAKE_BINARY_DIR}/generated)



include_directories( include )

add_subdirectory( test )

