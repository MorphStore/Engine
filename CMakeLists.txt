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

IF(CMAKE_BUILD_TYPE MATCHES Debug)
	morph_flag(-g)
	message(STATUS "MorphStore is configured in DEBUG mode.")
ELSEIF(CMAKE_BUILD_TYPE MATCHES Release)
	morph_flag(-O2)
	message(STATUS "MorphStore is configured in RELEASE mode.")
ELSEIF(CMAKE_BUILD_TYPE MATCHES HighPerf)
	morph_flag(-O3)
	morph_flag(-flto)
	message(STATUS "MorphStore is configured in HIGH PERFORMANCE mode.")
ELSEIF(CMAKE_BUILD_TYPE MATCHES Tally)
	morph_flag(-Dtally)
ELSE(CMAKE_BUILD_TYPE MATCHES Debug)
	message( SEND_ERROR "No known build type specified. Use either Debug, Release or HighPerf" )
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

IF(CSSE)
	morph_flag(-DSSE)
	morph_flag(-msse4.2)
	message(STATUS "SSE4.2 support switched on")
ENDIF(CSSE)

IF(CAVXTWO)
  morph_flag(-DSSE)
	morph_flag(-DAVXTWO)
	morph_flag(-mavx2)
	message(STATUS "AVX2 support switched on")
ENDIF(CAVXTWO)

IF(CAVX512)
  morph_flag(-DSSE)
	morph_flag(-DAVX512)
	morph_flag(-mavx512f)
	morph_flag(-mavx512pf)
	morph_flag(-mavx512er)
	morph_flag(-mavx512cd)
	morph_flag(-mavx512vl)
	morph_flag(-DAVXTWO)
	morph_flag(-mavx2)
	message(STATUS "AVX512 and AVX2 support switched on")
ENDIF(CAVX512)

IF(CNEON)
        morph_flag(-DNEON)
	morph_flag(-mfpu=neon)
	morph_flag(-flax-vector-conversions)
        message(STATUS "NEON support switched on")
ENDIF(CNEON)

IF(CRAPL)
    morph_flag(-DRAPL)
ENDIF(CRAPL)

IF(CODROID)
    morph_flag(-DODROID)
ENDIF(CODROID)

morph_flag(-march=native)

# remove build type to allow for custom flag handling
set(CMAKE_BUILD_TYPE "")

# general compiler settings, meant for all subdirectories and tests
morph_flag(-Werror)
morph_flag(-pedantic)
morph_flag(-Wall)
morph_flag(-Wextra)
morph_flag(-Wno-ignored-attributes)
morph_flag(-Wno-comment)
morph_flag(-fstack-protector-all)
morph_flag(-fno-tree-vectorize)

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

if ( DEFINED VBP_LIMIT_ROUTINES_FOR_SSB_SF1 )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tVBP_LIMIT_ROUTINES_FOR_SSB_SF1 is set to to TRUE\n" )
	morph_flag("-DVBP_LIMIT_ROUTINES_FOR_SSB_SF1")
	set( ignoreMe ${VBP_LIMIT_ROUTINES_FOR_SSB_SF1} )
else ( DEFINED VBP_LIMIT_ROUTINES_FOR_SSB_SF1 )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tVBP_LIMIT_ROUTINES_FOR_SSB_SF1 is set to FALSE\n" )
	morph_flag("-UVBP_LIMIT_ROUTINES_FOR_SSB_SF1")
endif ( DEFINED VBP_LIMIT_ROUTINES_FOR_SSB_SF1 )

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

execute_process(
        COMMAND python3 vbp_routine_gen.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/include/core/morphing
        OUTPUT_FILE vbp_routines.h
        RESULT_VARIABLE retValVBPRoutineGen
)
if(retValVBPRoutineGen EQUAL "1")
    message( FATAL_ERROR "Generation of the routines for vertical bit packing (vbp_l) failed.")
endif()

include_directories(${CMAKE_BINARY_DIR}/generated)

include_directories( include )

if( BUILD_ALL OR BUILD_CALIB OR BUILD_EXAMPLES OR BUILD_MICROBMS OR BUILD_SSB )
    add_subdirectory( src )
endif( BUILD_ALL OR BUILD_CALIB OR BUILD_EXAMPLES OR BUILD_MICROBMS OR BUILD_SSB )

if ( RUN_CTESTS )
	MESSAGE( "Running CTest has been enabled" )
	enable_testing()
  if ( CTEST_ALL OR CTEST_VECTOR OR CTEST_UTILS OR CTEST_STORAGE OR CTEST_PERSISTENCE OR CTEST_OPERATORS OR CTEST_MORPHING OR CTEST_MEMORY )
      add_subdirectory( test )
  endif( CTEST_ALL OR CTEST_VECTOR OR CTEST_UTILS OR CTEST_STORAGE OR CTEST_PERSISTENCE OR CTEST_OPERATORS OR CTEST_MORPHING OR CTEST_MEMORY )
endif( RUN_CTESTS )
