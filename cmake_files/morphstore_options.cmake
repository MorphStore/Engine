if(DEFINED CRAPL)
    add_compile_definitions(RAPL)
endif()

if(DEFINED CODROID)
    add_compile_definitions(ODROID)
endif()

########### <Checking Defines
set( MorphStoreProjectConf "MorphStore Project Configuration:\n" )

if ( DEFINED NO_SELF_MANAGING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMorphStore uses STANDARD C MALLOC.\n" )
	add_compile_definitions("MSV_NO_SELFMANAGED_MEMORY")
	set( ignoreMe ${NO_SELF_MANAGING} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMorphStore uses its CUSTOM MEMORY MANAGER.\n" )
	add_definitions("-UMSV_NO_SELFMANAGED_MEMORY")
endif ()

if ( DEFINED DEBUG_MALLOC )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager calls will be printed to the debug channel.\n" )
	add_compile_definitions("MSV_DEBUG_MALLOC")
	set( ignoreMe ${DEBUG_MALLOC} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager calls are not printed.\n")
	add_definitions("-UMSV_DEBUG_MALLOC")
endif ()

if ( DEFINED CHECK_LEAKING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager will check for potential leaks.\n" )
	add_compile_definitions("MSV_MEMORY_LEAK_CHECK")
	set( ignoreMe ${CHECK_LEAKING} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager is oblivious to potential memory leaks.\n" )
	add_definitions("-UMSV_MEMORY_LEAK_CHECK")
endif ()

if ( DEFINED MMGR_ALIGN )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager aligns to ${MMGR_ALIGN} Byte.\n" )
	add_compile_definitions("MSV_MEMORY_MANAGER_ALIGNMENT_BYTE=${MMGR_ALIGN}")
	set( ignoreMe ${MMGR_ALIGN} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tMemoryManager align defaults to 64 Byte.\n" )
	add_definitions("-DMSV_MEMORY_MANAGER_ALIGNMENT_BYTE=64")
endif ()

if ( DEFINED NOLOGGING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tLogging is completely disabled.\n")
	add_compile_definitions("MSV_NO_LOG" )
	set( ignoreMe ${NOLOGGING} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tLogging is set to standard.\n")
	add_definitions("-UMSV_NO_LOG" )
endif ()

if ( DEFINED QMMMES )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tQueryMemoryManager min expand size is set to ${QMMMES}\n" )
	add_compile_definitions("MSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE=${QMMMES}")
	set( ignoreMe ${QMMMES} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tQueryMemoryManager min expand size defaults to 128M.\n" )
	add_definitions("-UMSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE")
endif ()

if ( DEFINED QMMIS )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tQueryMemoryManager initial size is set to ${QMMIS}\n" )
	add_compile_definitions("MSV_QUERY_MEMORY_MANAGER_INITIAL_SIZE=${QMMIS}")
	set( ignoreMe ${QMMIS} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tQueryMemoryManager initial size defaults to 128M.\n" )
	add_definitions("-UMSV_QUERY_MEMORY_MANAGER_INITIAL_SIZE")
endif ()

if ( DEFINED QMMAE )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tQueryMemoryManager is allowed to expand its size.\n" )
	add_compile_definitions("MSV_QUERY_MEMORY_MANAGER_ALLOW_EXPAND")
	set( ignoreMe ${QMMAE} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tQueryMemoryManager is NOT allowed to expand its size.\n" )
	add_definitions("-UMSV_QUERY_MEMORY_MANAGER_ALLOW_EXPAND")
endif ()

if ( DEFINED QMMIB )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tQueryMemoryManager will initialize buffers after allocation.\n" )
	add_compile_definitions("MSV_QUERY_MEMORY_MANAGER_INITIALIZE_BUFFERS")
	set( ignoreMe ${QMMIB} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tQueryMemoryManager will NOT initialize buffers after allocation.\n" )
	add_definitions("-UMSV_QUERY_MEMORY_MANAGER_INITIALIZE_BUFFERS")
endif ()

if ( DEFINED ENABLE_MONITORING )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tENABLE_MONITORING is set to to TRUE\n" )
	add_compile_definitions("MSV_USE_MONITORING")
	set( ignoreMe ${ENABLE_MONITORING} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tENABLE_MONITORING is set to FALSE\n" )
	add_definitions("-UMSV_USE_MONITORING")
endif ()

if ( DEFINED VBP_LIMIT_ROUTINES_FOR_SSB_SF1 )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tVBP_LIMIT_ROUTINES_FOR_SSB_SF1 is set to to TRUE\n" )
	add_compile_definitions("VBP_LIMIT_ROUTINES_FOR_SSB_SF1")
	set( ignoreMe ${VBP_LIMIT_ROUTINES_FOR_SSB_SF1} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tVBP_LIMIT_ROUTINES_FOR_SSB_SF1 is set to FALSE\n" )
	add_definitions("-UVBP_LIMIT_ROUTINES_FOR_SSB_SF1")
endif ()

if ( DEFINED VBP_LIMIT_ROUTINES_FOR_SSB_SF100 )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tVBP_LIMIT_ROUTINES_FOR_SSB_SF100 is set to to TRUE\n" )
	add_compile_definitions("VBP_LIMIT_ROUTINES_FOR_SSB_SF100")
	set( ignoreMe ${VBP_LIMIT_ROUTINES_FOR_SSB_SF100} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tVBP_LIMIT_ROUTINES_FOR_SSB_SF100 is set to FALSE\n" )
	add_definitions("-UVBP_LIMIT_ROUTINES_FOR_SSB_SF100")
endif ()

if ( DEFINED PROJECT_ASSUME_PREPARED )
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tPROJECT_ASSUME_PREPARED is set to to TRUE\n" )
	add_compile_definitions("PROJECT_ASSUME_PREPARED")
	set( ignoreMe ${PROJECT_ASSUME_PREPARED} )
else ()
	set( MorphStoreProjectConf "${MorphStoreProjectConf}\tPROJECT_ASSUME_PREPARED is set to FALSE\n" )
	add_definitions("-UPROJECT_ASSUME_PREPARED")
endif ()

message( ${MorphStoreProjectConf} )
file( WRITE ${LOG_FILE} ${MorphStoreProjectConf} )

########### Checking Defines />

