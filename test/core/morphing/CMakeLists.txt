if ( CTEST_ALL OR CTEST_MORPHING )
	#FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/delta_test_app )
	#FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/group_simple_test_app )
	#FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/k_wise_ns_test_app )
	#FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/vbp_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/IR_transformation_uncompr_test_app )
	#FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/IR_transformation_compr_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/bitmask_morph_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/position_list_morph_test_app )

	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/write_iterator_IR_pl_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/write_iterator_IR_bm_test_app )
	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/write_iterator_IR_wah_compr_test_app )

	FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/morphing/wah_bm_test_app )

	#add_executable( delta_test_app delta_test.cpp )
	#add_executable( group_simple_test_app group_simple_test.cpp )
	#add_executable( k_wise_ns_test_app k_wise_ns_test.cpp )  #
	#add_executable( vbp_test_app vbp_test.cpp )
	add_executable( IR_transformation_uncompr_test_app DA_bitmap_IR/IR_transformation_uncompr_test.cpp)
	#add_executable( IR_transformation_compr_test_app IR_transformation_compr_test.cpp)
	add_executable( bitmask_morph_test_app DA_bitmap_IR/bm_compr_test.cpp)
	add_executable( position_list_morph_test_app DA_bitmap_IR/pl_compr_test.cpp)

	add_executable( write_iterator_IR_pl_test_app DA_bitmap_IR/write_iterator_IR_pl_test.cpp)
	add_executable( write_iterator_IR_bm_test_app DA_bitmap_IR/write_iterator_IR_bm_test.cpp)
	add_executable( write_iterator_IR_wah_compr_test_app DA_bitmap_IR/write_iterator_IR_wah_compr_test.cpp)

	add_executable( wah_bm_test_app DA_bitmap_IR/wah_bm_test.cpp)

	#target_compile_options( delta_test_app PRIVATE
	# space to add custom flags for THIS SPECIFIC TARGET
	#)
	#target_compile_options( group_simple_test_app PRIVATE
	# space to add custom flags for THIS SPECIFIC TARGET
	#)
	#target_compile_options( k_wise_ns_test_app PRIVATE
	# space to add custom flags for THIS SPECIFIC TARGET
	#)
	#target_compile_options( vbp_test_app PRIVATE
	# space to add custom flags for THIS SPECIFIC TARGET
	#)
	target_compile_options( IR_transformation_uncompr_test_app PRIVATE
			# space to add custom flags for THIS SPECIFIC TARGET
	)
	#target_compile_options( IR_transformation_compr_test_app PRIVATE
			# space to add custom flags for THIS SPECIFIC TARGET
	#)
	target_compile_options( bitmask_morph_test_app PRIVATE
			# space to add custom flags for THIS SPECIFIC TARGET
	)
	target_compile_options( position_list_morph_test_app PRIVATE
			# space to add custom flags for THIS SPECIFIC TARGET
	)

	target_compile_options( write_iterator_IR_pl_test_app PRIVATE
			# space to add custom flags for THIS SPECIFIC TARGET
	)

	target_compile_options( write_iterator_IR_bm_test_app PRIVATE
			# space to add custom flags for THIS SPECIFIC TARGET
	)

	target_compile_options( write_iterator_IR_wah_compr_test_app PRIVATE
			# space to add custom flags for THIS SPECIFIC TARGET
	)

	target_compile_options( wah_bm_test_app PRIVATE
			# space to add custom flags for THIS SPECIFIC TARGET
	)

	#target_link_libraries( delta_test_app PRIVATE "-ldl" )
	#target_link_libraries( group_simple_test_app PRIVATE "-ldl" )
	#target_link_libraries( k_wise_ns_test_app PRIVATE "-ldl" )
	#target_link_libraries( vbp_test_app PRIVATE "-ldl" )
	target_link_libraries( IR_transformation_uncompr_test_app PRIVATE "-ldl" )
	#target_link_libraries( IR_transformation_compr_test_app PRIVATE "-ldl" )
	target_link_libraries( bitmask_morph_test_app PRIVATE "-ldl" )
	target_link_libraries( position_list_morph_test_app PRIVATE "-ldl" )

	target_link_libraries( write_iterator_IR_pl_test_app PRIVATE "-ldl" )
	target_link_libraries( write_iterator_IR_bm_test_app PRIVATE "-ldl" )
	target_link_libraries( write_iterator_IR_wah_compr_test_app PRIVATE "-ldl" )

	target_link_libraries( wah_bm_test_app PRIVATE "-ldl" )

	#add_test( delta_test delta_test_app )
	#add_test( group_simple_test group_simple_test_app )
	#add_test( k_wise_ns_test k_wise_ns_test_app )
	#add_test( vbp_test vbp_test_app )
	add_test( IR_transformation_uncompr_test IR_transformation_uncompr_test_app )
	#add_test( IR_transformation_compr_test IR_transformation_compr_test_app )
	add_test( bitmask_morph_test bitmask_morph_test_app )
	add_test( position_list_morph_test position_list_morph_test_app )

	add_test( write_iterator_IR_pl_test write_iterator_IR_pl_test_app )
	add_test( write_iterator_IR_bm_test write_iterator_IR_bm_test_app )
	add_test( write_iterator_IR_wah_compr_test write_iterator_IR_wah_compr_test_app )

	add_test( wah_bm_test wah_bm_test_app )
endif()