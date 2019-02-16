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
set( MorphStoreProjectConf "MorphStore Project Configuration:\n" )

if ( DEFINED NO_SELF_MANAGING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMorphStore uses STANDARD C MALLOC.\n" )
	add_definitions("-DMSV_NO_SELFMANAGED_MEMORY")
	set( ignoreMe ${NO_SELF_MANAGING} )
else ( DEFINED NO_SELF_MANAGING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMorphStore uses its CUSTOM MEMORY MANAGER.\n" )
	add_definitions("-UMSV_NO_SELFMANAGED_MEMORY")
endif ( DEFINED NO_SELF_MANAGING )

if ( DEFINED DEBUG_MALLOC )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager calls will be printed to the debug channel.\n" )
	add_definitions("-DMSV_DEBUG_MALLOC")
	set( ignoreMe ${DEBUG_MALLOC} )
else ( DEFINED DEBUG_MALLOC )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager calls are not printed.\n")
	add_definitions("-UMSV_DEBUG_MALLOC")
endif ( DEFINED DEBUG_MALLOC )

if ( DEFINED CHECK_LEAKING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager will check for potential leaks.\n" )
	add_definitions("-DMSV_MEMORY_LEAK_CHECK")
	set( ignoreMe ${CHECK_LEAKING} )
else ( DEFINED CHECK_LEAKING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager is oblivious to potential memory leaks.\n" )
	add_definitions("-UMSV_MEMORY_LEAK_CHECK")
endif ( DEFINED CHECK_LEAKING )

if ( DEFINED MMGR_ALIGN )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager aligns to ${MMGR_ALIGN} Byte.\n" )
	add_definitions("-DMSV_MEMORY_MANAGER_ALIGNMENT_BYTE=${MMGR_ALIGN}")
	set( ignoreMe ${MMGR_ALIGN} )
else ( DEFINED MMGR_ALIGN )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager align defaults to 64 Byte.\n" )
	add_definitions("-DMSV_MEMORY_MANAGER_ALIGNMENT_BYTE=64")
endif ( DEFINED MMGR_ALIGN )

if ( DEFINED NOLOGGING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tLogging is completely disabled.\n")
	add_definitions("-DMSV_NO_LOG" )
	set( ignoreMe ${NOLOGGING} )
else ( DEFINED NOLOGGING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tLogging is set to standard.\n")
	add_definitions("-UMSV_NO_LOG" )
endif ( DEFINED NOLOGGING )

if ( DEFINED QMMMES )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tQMMMES is set to ${QMMMES}\n" )
	add_definitions("-DMSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE=${QMMMES}")
	set( ignoreMe ${QMMMES} )
else ( DEFINED QMMMES )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tQMMMES defaults to 128M.\n" )
	add_definitions("-UMSV_NO_SELFMANAGED_MEMORY")
endif ( DEFINED QMMMES )

MESSAGE( ${MorphStoreProjectConf} )
file( WRITE recentMorphStoreProjectConf.log ${MorphStoreProjectConf} )

########### Checking Defines />


add_definitions("-DGIT_COMMIT_HASH=${GIT_COMMIT_HASH}")
add_definitions("-DGIT_BRANCH=${GIT_BRANCH}")

configure_file(
   ${CMAKE_SOURCE_DIR}/include/core/utils/cmake_template.h.in
   ${CMAKE_BINARY_DIR}/generated/cmake_template.h
)

include_directories(${CMAKE_BINARY_DIR}/generated)



include_directories( include )

add_subdirectory( src )
add_subdirectory( test )

