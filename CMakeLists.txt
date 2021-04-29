cmake_minimum_required( VERSION 3.10 )
project( MorphStore CXX )

########################################################################################################################
####### include local cmake config #####################################################################################
########################################################################################################################
if( NOT EXISTS ${PROJECT_SOURCE_DIR}/local_config.cmake)
    file(WRITE ${PROJECT_SOURCE_DIR}/local_config.cmake "## This is your local cmake config file, to preset certain cmake variables. Do not upload into the git repository!\n")
endif()
include( ${PROJECT_SOURCE_DIR}/local_config.cmake OPTIONAL )
message( STATUS "You can define your own local cmake options/variables inside the local_config.cmake file in the root directory of the project." )

########################################################################################################################
####### set important variables ########################################################################################
########################################################################################################################
####### README:
####### The check if a variable is already defined will enable to import MorphStore as a submodule without overwriting
####### already set variables
########################################################################################################################

####### set c++ version
if(NOT DEFINED CMAKE_CXX_STANDARD)
	set( CMAKE_CXX_STANDARD 20 )
elseif(${CMAKE_CXX_STANDARD} LESS 17)
	message( FATAL_ERROR "Extern defined C++ standard is to old. C++17 or above is required.")
endif()

####### set root path of MorphStore project
if(NOT DEFINED MorphStoreRoot)
	set( MorphStoreRoot ${PROJECT_SOURCE_DIR})
endif()

####### set path to log file, where the last morphstore config is printed to
set( LOG_FILE "${MorphStoreRoot}/recentMorphStoreProjectConf.log" )



########################################################################################################################
####### import user defined cmake functions ############################################################################
########################################################################################################################
####### import function generate_git_hash() // fetches and writes git branch and commit hash into <build_path>/generated/cmake_template.h
include( ${MorphStoreRoot}/cmake_files/generate_git_hash.cmake )

####### import function generate_vbp_routine_code() // generates code for vertical bit packing algorithms
include( ${MorphStoreRoot}/cmake_files/generate_vbp_routine_code.cmake )

## @todo throws "fatal: bad object HEAD" when building remote (because git files are not uploaded to server)
generate_git_hash()

if( NOT EXISTS "${MorphStoreRoot}/include/core/morphing/vbp_routines.h" OR DEFINED REGENERATE_VBP_ROUTINES )
    generate_vbp_routine_code()
endif()



########################################################################################################################
####### import user defined cmake functions ############################################################################
########################################################################################################################
####### README:
#######
########################################################################################################################

####### set build type dependend compile options // valid types are: Debug, Release, HighPerf
include( ${MorphStoreRoot}/cmake_files/build_type.cmake )

####### set vector extension specific compilter options
include( ${MorphStoreRoot}/cmake_files/vector_extension_options.cmake )

####### set general compilter options
include( ${MorphStoreRoot}/cmake_files/general_compiler_options.cmake )

####### set further morphstore options
include( ${MorphStoreRoot}/cmake_files/morphstore_options.cmake )




########################################################################################################################
####### set include directories ########################################################################################
########################################################################################################################

####### @todo does this make any sense, when TVL files are still part of the source code!?
if ( DEFINED TVL_PATH)
  include_directories(AFTER ${TVL_PATH}  ${PROJECT_SOURCE_DIR}/include)
else ()
  include_directories( ${PROJECT_SOURCE_DIR}/include )
endif()

####### for cmake_template.h containing git info
include_directories(${CMAKE_BINARY_DIR}/generated)

####### shortcut to include helper
include_directories(${PROJECT_SOURCE_DIR}/include/abridge)


########################################################################################################################
####### traverse through subdirectories ################################################################################
########################################################################################################################
add_subdirectory( src )

########################################################################################################################
####### run tests ######################################################################################################
########################################################################################################################
## @todo Rework?
if ( RUN_CTESTS )
	MESSAGE( "Running CTest has been enabled" )
	enable_testing()
  if ( CTEST_ALL OR CTEST_VECTOR OR CTEST_UTILS OR CTEST_STORAGE OR CTEST_PERSISTENCE OR CTEST_OPERATORS OR CTEST_MORPHING OR CTEST_MEMORY OR CTEST_VIRTUAL )
      add_subdirectory( test )
  endif( CTEST_ALL OR CTEST_VECTOR OR CTEST_UTILS OR CTEST_STORAGE OR CTEST_PERSISTENCE OR CTEST_OPERATORS OR CTEST_MORPHING OR CTEST_MEMORY OR CTEST_VIRTUAL )
endif( RUN_CTESTS )
