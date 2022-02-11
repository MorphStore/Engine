macro(generate_vbp_routine_code)
    execute_process(
            COMMAND python3 group_simple_routine_gen.py
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/include/core/morphing
            OUTPUT_FILE group_simple_routines.h
            RESULT_VARIABLE retValGroupSimpleRoutineGen
    )
    if(retValGroupSimpleRoutineGen EQUAL "1")
        message( FATAL_ERROR "Generation of the routines for SIMD-Group-Simple (group_simple_f) failed.")
    endif()
    execute_process(
        COMMAND python3 vbp_routine_gen.py
        WORKING_DIRECTORY ${MorphStoreRoot}/include/core/morphing
        OUTPUT_FILE vbp_routines.h
        RESULT_VARIABLE retValVBPRoutineGen
    )
    if(retValVBPRoutineGen EQUAL "1")
        message( FATAL_ERROR "Generation of the routines for vertical bit packing (vbp_l) failed.")
    endif()
endmacro()
