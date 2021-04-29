if(CMAKE_BUILD_TYPE MATCHES Debug)
	add_compile_options(-g)
	message(STATUS "MorphStore is configured in DEBUG mode.")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
	add_compile_options(-O2)
	message(STATUS "MorphStore is configured in RELEASE mode.")
elseif(CMAKE_BUILD_TYPE MATCHES HighPerf)
	add_compile_options(-O3)
	add_compile_options(-flto)
	message(STATUS "MorphStore is configured in HIGH PERFORMANCE mode.")
else(CMAKE_BUILD_TYPE MATCHES Debug)
	message( FATAL_ERROR "No known build type specified. Use either Debug, Release or HighPerf" )
endif()

set(BUILD_TYPE ${CMAKE_BUILD_TYPE})

# remove build type to allow for custom flag handling
set(CMAKE_BUILD_TYPE "")
