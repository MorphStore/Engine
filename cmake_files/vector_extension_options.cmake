if(DEFINED CSSE)
	add_compile_definitions(SSE)
	morph_flag(-msse4.2)
	message(STATUS "SSE4.2 support switched on")
endif()

if(DEFINED CAVXTWO)
  	add_compile_definitions(SSE)
	add_compile_definitions(AVXTWO)
	add_compile_options(-mavx2)
	message(STATUS "AVX2 support switched on")
endif()

if(DEFINED CAVX512)
  	add_compile_definitions(SSE)
	add_compile_definitions(AVXTWO)
	add_compile_definitions(AVX512)
	add_compile_options(-mavx512f)
	add_compile_options(-mavx512pf)
	add_compile_options(-mavx512er)
	add_compile_options(-mavx512cd)
	add_compile_options(-mavx512vl)
	add_compile_options(-mavx2)
	message(STATUS "AVX512 and AVX2 support switched on")
endif()

if(DEFINED CNEON)
  	add_compile_definitions(NEON)
	add_compile_options(-mfpu=neon)
	add_compile_options(-flax-vector-conversions)
  	message(STATUS "NEON support switched on")
endif()

if(DEFINED CSVE)
  add_compile_definitions(SVE)
  add_compile_options(-march=armv8-a+sve)
  message(STATUS "SVE support switched on")
else()
  add_compile_options(-march=native)
endif()