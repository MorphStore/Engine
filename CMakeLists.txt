cmake_minimum_required( VERSION 3.10 )
project( MorphStore )

set( CMAKE_CXX_STANDARD 14 )

macro(morph_flag)
	add_definitions(${ARGN})
	LIST(APPEND MODULE_FLAGS ${ARGN})
endmacro(morph_flag)

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

set( LOG_FILE "recentMorphStoreProjectConf.log" )

if ( ${CARCHITECTURE} MATCHES "X86" )
	include( CMake_GCC_x86.cmake )
elseif ( ${CARCHITECTURE} MATCHES "TSUBASA" )
	if ( ${CCOMPILER} MATCHES "CLANG" )
#		list(APPEND CMAKE_PREFIX_PATH "/opt/nec/nosupport/llvm-ve/")
#		find_package(LLVM REQUIRED CONFIG)
#
#		message(STATUS "#######Found LLVM ${LLVM_PACKAGE_VERSION}")
#		message(STATUS "#######Using LLVMConfig.cmake in: ${LLVM_DIR}")
#		include_directories(${LLVM_INCLUDE_DIRS})
#		add_definitions(${LLVM_DEFINITIONS})
#
#		find_package(Clang REQUIRED CONFIG)

		#		set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "")
#		set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "")
		include( CMake_LLVM_tsubasa.cmake )
	elseif ( ${CCOMPILER} MATCHES "NCC" )
		include( CMake_NCC_tsubasa.cmake )
	endif ( ${CCOMPILER} MATCHES "CLANG" )
endif ( ${CARCHITECTURE} MATCHES "X86" )
############################################################

message( STATUS "###       LINKER_EXEC: ${CMAKE_CXX_LINK_EXECUTABLE}")
message( STATUS "###            LINKER: ${CMAKE_LINKER}")
message( STATUS "###             FLAGS: ${CMAKE_LINKER_FLAGS}")
message( STATUS "###      CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message( STATUS "###                AR: ${CMAKE_AR}")
message( STATUS "###            RANLIB: ${CMAKE_RANLIB}")
message( STATUS "###     LINK_IMPLICIT: ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES}")
message( STATUS "### LINK_IMPLICIT_DIR: ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}")
message("LLVM_INCLUDE_DIRS=${LLVM_INCLUDE_DIRS}")
message("LLVM_DEFINITIONS=${LLVM_DEFINITIONS}")

# general compiler settings, meant for all subdirectories and tests
add_general_compiler_flags()
add_analysis_compiler_flags()
add_optimization_compiler_flags()
add_vectorization_flags()

# remove build type to allow for custom flag handling
set(CMAKE_BUILD_TYPE "")

########### <Checking Defines
set( MorphStoreProjectConf "MorphStore Project Configuration:\n" )

if ( DEFINED NO_SELF_MANAGING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMorphStore uses STANDARD C MALLOC.\n" )
	morph_flag("-DMSV_NO_SELFMANAGED_MEMORY")
	set( ignoreMe ${NO_SELF_MANAGING} )
else ( DEFINED NO_SELF_MANAGING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMorphStore uses its CUSTOM MEMORY MANAGER.\n" )
	morph_flag("-UMSV_NO_SELFMANAGED_MEMORY")
endif ( DEFINED NO_SELF_MANAGING )

if ( DEFINED DEBUG_MALLOC )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager calls will be printed to the debug channel.\n" )
	morph_flag("-DMSV_DEBUG_MALLOC")
	set( ignoreMe ${DEBUG_MALLOC} )
else ( DEFINED DEBUG_MALLOC )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager calls are not printed.\n")
	morph_flag("-UMSV_DEBUG_MALLOC")
endif ( DEFINED DEBUG_MALLOC )

if ( DEFINED CHECK_LEAKING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager will check for potential leaks.\n" )
	morph_flag("-DMSV_MEMORY_LEAK_CHECK")
	set( ignoreMe ${CHECK_LEAKING} )
else ( DEFINED CHECK_LEAKING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager is oblivious to potential memory leaks.\n" )
	morph_flag("-UMSV_MEMORY_LEAK_CHECK")
endif ( DEFINED CHECK_LEAKING )

if ( DEFINED MMGR_ALIGN )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager aligns to ${MMGR_ALIGN} Byte.\n" )
	morph_flag("-DMSV_MEMORY_MANAGER_ALIGNMENT_BYTE=${MMGR_ALIGN}")
	set( ignoreMe ${MMGR_ALIGN} )
else ( DEFINED MMGR_ALIGN )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager align defaults to 64 Byte.\n" )
	morph_flag("-DMSV_MEMORY_MANAGER_ALIGNMENT_BYTE=64")
endif ( DEFINED MMGR_ALIGN )

if ( DEFINED NOLOGGING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tLogging is completely disabled.\n")
	morph_flag("-DMSV_NO_LOG" )
	set( ignoreMe ${NOLOGGING} )
else ( DEFINED NOLOGGING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tLogging is set to standard.\n")
	morph_flag("-UMSV_NO_LOG" )
endif ( DEFINED NOLOGGING )

if ( DEFINED QMMMES )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tQMMMES is set to ${QMMMES}\n" )
	morph_flag("-DMSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE=${QMMMES}")
	set( ignoreMe ${QMMMES} )
else ( DEFINED QMMMES )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tQMMMES defaults to 128M.\n" )
	morph_flag("-UMSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE")
endif ( DEFINED QMMMES )

if ( DEFINED ENABLE_MONITORING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tENABLE_MONITORING is set to to TRUE\n" )
	morph_flag("-DMSV_USE_MONITORING")
	set( ignoreMe ${ENABLE_MONITORING} )
else ( DEFINED ENABLE_MONITORING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tENABLE_MONITORING is set to FALSE\n" )
	morph_flag("-UMSV_USE_MONITORING")
endif ( DEFINED ENABLE_MONITORING )

MESSAGE( ${MorphStoreProjectConf} )
file( WRITE ${LOG_FILE} ${MorphStoreProjectConf} )

########### Checking Defines />

include_directories( ${PROJECT_SOURCE_DIR}/include )

morph_flag("-DGIT_COMMIT_HASH=${GIT_COMMIT_HASH}")
morph_flag("-DGIT_BRANCH=${GIT_BRANCH}")

configure_file(
   ${CMAKE_SOURCE_DIR}/include/core/utils/cmake_template.h.in
   ${CMAKE_BINARY_DIR}/generated/cmake_template.h
)

include_directories(${CMAKE_BINARY_DIR}/generated)

#include_directories( include )

#add_subdirectory( src )

if ( RUN_CTESTS )
	MESSAGE( "Running CTest has been enabled" )
	enable_testing()
  if ( CTEST_ALL )
      add_subdirectory( src )
  elseif ( CTEST_QUERIES )
          add_subdirectory( src/examples )
  endif( CTEST_ALL )
  if ( CTEST_ALL OR CTEST_VECTOR OR CTEST_UTILS OR CTEST_STORAGE OR CTEST_PERSISTENCE OR CTEST_OPERATORS OR CTEST_MORPHING OR CTEST_MEMORY )
      add_subdirectory( test )
  endif( CTEST_ALL OR CTEST_VECTOR OR CTEST_UTILS OR CTEST_STORAGE OR CTEST_PERSISTENCE OR CTEST_OPERATORS OR CTEST_MORPHING OR CTEST_MEMORY )
endif( RUN_CTESTS )