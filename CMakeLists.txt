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

########### <Checking Defines
MESSAGE( "Project configuration:")

if ( DEFINED NO_SELF_MANAGING )
	MESSAGE("\tMorphStore uses STANDARD C MALLOC.")
	add_definitions("-DMSV_NO_SELFMANAGED_MEMORY")
	set( ignoreMe NO_SELF_MANAGING )
else ( DEFINED NO_SELF_MANAGING )
	MESSAGE("\tMorphStore uses its CUSTOM MEMORY MANAGER.")
	add_definitions("-UMSV_NO_SELFMANAGED_MEMORY")
endif ( DEFINED NO_SELF_MANAGING )

if ( DEFINED NOLOGGING )
	MESSAGE("\tLogging is completely disabled.")
	add_definitions("-DMSV_NO_LOG" )
	set( ignoreMe NOLOGGING )
else ( DEFINED NOLOGGING )
	MESSAGE("\tLogging is set to standard.")
	add_definitions("-UMSV_NO_LOG" )
endif ( DEFINED NOLOGGING )

if ( DEFINED DEBUG_MALLOC )
	MESSAGE("\tMemoryManager calls will be printed to the debug channel.")
	add_definitions("-DMSV_DEBUG_MALLOC")
	set( ignoreMe DEBUG_MALLOC )
else ( DEFINED DEBUG_MALLOC )
	MESSAGE("\tMemoryManager calls are not printed.")
	add_definitions("-UMSV_DEBUG_MALLOC")
endif ( DEFINED DEBUG_MALLOC )

if ( DEFINED QMMMES )
	MESSAGE( "\tQMMMES is set to ${QMMMES}" )
	add_definitions("-DMSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE=${QMMMES}")
	set( ignoreMe QMMMES )
else ( DEFINED QMMMES )
	MESSAGE( "\tQMMMES defaults to 128M." )
	add_definitions("-UMSV_NO_SELFMANAGED_MEMORY")
endif ( DEFINED QMMMES )
########### Checking Defines />


add_definitions("-DGIT_COMMIT_HASH=${GIT_COMMIT_HASH}")
add_definitions("-DGIT_BRANCH=${GIT_BRANCH}")

configure_file(
   ${CMAKE_SOURCE_DIR}/include/core/utils/cmake_template.h.in
   ${CMAKE_BINARY_DIR}/generated/cmake_template.h
)

include_directories(${CMAKE_BINARY_DIR}/generated)



include_directories( include )

add_subdirectory( test )

