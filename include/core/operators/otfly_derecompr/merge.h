/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/

/**
 * @file merge_uncompr.h
 * @brief TODO
 */

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_MERGE_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_MERGE_UNCOMPR_H


#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <cassert>

namespace morphstore {
	
	using namespace vectorlib;
	
	template< class VectorExtension >
	struct merge_sorted_processing_unit {
		IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
		
		struct state_t {
			vector_mask_t m_MaskGreater;//Do we really need this as a state?
			int m_doneLeft;
			int m_doneRight;
			
			//vector_t m_RotVec;
			state_t(void) : m_MaskGreater{(vector_mask_t) 0} {}
			
			state_t(vector_mask_t const & p_MaskGreater) : m_MaskGreater{p_MaskGreater} {}
			
			
		};
		
		MSV_CXX_ATTRIBUTE_FORCE_INLINE
		static
		vector_mask_t
		apply(
		  vector_t const p_Data1Vector,
		  vector_t const p_Data2Vector,
		  
		  state_t & p_State
		) {
			vector_mask_t resultMaskEqual = 0;
			
			resultMaskEqual = vectorlib::equal<VectorExtension>::apply(p_Data2Vector, p_Data1Vector);
			p_State.m_MaskGreater = vectorlib::greater<VectorExtension>::apply(
			  p_Data1Vector,
			  p_Data2Vector
			);// vec2<vec1?
			
			
			return resultMaskEqual;
		}
	};
	
	template< class VectorExtension >
	struct merge_sorted_batch {
		IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
		
		MSV_CXX_ATTRIBUTE_FORCE_INLINE static size_t apply(
		  base_t * p_Data1Ptr,//left
		  base_t * p_Data2Ptr,//right
		  base_t * p_OutPtr,
		  size_t const p_CountData1,
		  size_t const p_CountData2,
		  typename merge_sorted_processing_unit<VectorExtension>::state_t & p_State
		) {
			
			//We hope that the larger column has the longest sequential runs, i.e. we get more sequential memory access
			base_t * pl = p_Data1Ptr;
			base_t * pr = p_Data2Ptr;
			
			base_t * endInPosR = p_Data2Ptr + p_CountData2;
			base_t * endInPosL = p_Data1Ptr + p_CountData1;
			
			//We hope that the larger column has the longest sequential runs, i.e. we get more sequential memory access
			if (p_CountData2 < p_CountData1) {
				
				p_Data1Ptr = pr;
				p_Data2Ptr = pl;
				endInPosR = p_Data2Ptr + p_CountData1;
				endInPosL = p_Data1Ptr + p_CountData2;
			}
			
			base_t const * out_init = p_OutPtr;
			
			
			vector_t data1Vector = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(*p_Data1Ptr);
			
			vector_t data2Vector = vectorlib::load<
			  VectorExtension,
			  vectorlib::iov::ALIGNED,
			  vector_size_bit::value
			>(p_Data2Ptr);
			
			
			while (p_Data2Ptr < (endInPosR - vector_element_count::value) && p_Data1Ptr < endInPosL) {
				
				vector_mask_t resultMaskEqual =
				  merge_sorted_processing_unit<VectorExtension>::apply(
					data1Vector,
					data2Vector,
					p_State
				  );
				
				
				if ((p_State.m_MaskGreater) == 0) {
					if (resultMaskEqual == 0) {
						*p_OutPtr = vectorlib::extract_value<VectorExtension, vector_base_t_granularity::value>(
						  data1Vector,
						  0
						);
						p_OutPtr ++;
					}
					p_Data1Ptr ++;
					data1Vector = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(*p_Data1Ptr);
				} else {
					vectorlib::compressstore<
					  VectorExtension,
					  vectorlib::iov::UNALIGNED,
					  vector_base_t_granularity::value
					>(p_OutPtr, data2Vector, p_State.m_MaskGreater);
					//p_Data2Ptr += __builtin_popcountl(p_State.m_MaskGreater);
					//p_OutPtr += __builtin_popcountl(p_State.m_MaskGreater);
					p_Data2Ptr += vectorlib::count_matches<VectorExtension>::apply(p_State.m_MaskGreater);
					p_OutPtr += vectorlib::count_matches<VectorExtension>::apply(p_State.m_MaskGreater);
					data2Vector = vectorlib::load<VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>(
					  p_Data2Ptr
					);
					
				}
			}
			
			
			//Remainder must be done sequentially
			while (p_Data1Ptr < endInPosL && p_Data2Ptr < endInPosR) {
				
				if (*p_Data1Ptr < *p_Data2Ptr) {
					*p_OutPtr = *p_Data1Ptr;
					p_Data1Ptr ++;
				} else if (*p_Data2Ptr < *p_Data1Ptr) {
					*p_OutPtr = *p_Data2Ptr;
					p_Data2Ptr ++;
				} else { // *inPosL == *inPosR
					*p_OutPtr = *p_Data1Ptr;
					p_Data1Ptr ++;
					p_Data2Ptr ++;
				}
				p_OutPtr ++;
			}
			
			
			while (p_Data1Ptr < (endInPosL - vector_element_count::value)) {
				data1Vector = vectorlib::load<VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>(
				  p_Data1Ptr
				);
				vectorlib::store<VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>(
				  p_OutPtr,
				  data1Vector
				);
				p_OutPtr += vector_element_count::value;
				p_Data1Ptr += vector_element_count::value;
				
			}
			
			while (p_Data2Ptr < (endInPosR - vector_element_count::value)) {
				
				data2Vector = vectorlib::load<VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>(
				  p_Data2Ptr
				);
				vectorlib::store<VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>(
				  p_OutPtr,
				  data2Vector
				);
				
				p_Data2Ptr += vector_element_count::value;
				p_OutPtr += vector_element_count::value;
				
			}
			
			//Copy rest, which didn't fit in a vetor register
			while (p_Data1Ptr < endInPosL) {
				*p_OutPtr = *p_Data1Ptr;
				p_Data1Ptr ++;
				p_OutPtr ++;
			}
			
			while (p_Data2Ptr < endInPosR) {
				*p_OutPtr = *p_Data2Ptr;
				p_Data2Ptr ++;
				p_OutPtr ++;
			}
			
			
			return (p_OutPtr - out_init);
		}
	};
	
	template< class t_vector_extension >
	struct merge_sorted_t<t_vector_extension, uncompr_f, uncompr_f, uncompr_f> {
		IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
		
		MSV_CXX_ATTRIBUTE_FORCE_INLINE
		static
		column <uncompr_f> const *
		apply(
		  column <uncompr_f> const * const in_L_posColumn,
		  column <uncompr_f> const * const in_R_posColumn
		
		) {
			const size_t in_L_dataCount = in_L_posColumn->get_count_values();
			const size_t in_R_dataCount = in_R_posColumn->get_count_values();
			//assert(inData1Count == p_Data2Column->get_count_values());
			
			base_t * in_L_pos = in_L_posColumn->get_data();
			base_t * in_R_pos = in_R_posColumn->get_data();
			
			size_t const sizeByte = in_L_posColumn->get_size_used_byte() + in_R_posColumn->get_size_used_byte();
			
			typename merge_sorted_processing_unit<t_vector_extension>::state_t vectorState;
			typename merge_sorted_processing_unit<scalar<v64<uint64_t>>>::state_t scalarState;
			
			auto out_posColumn = new column<uncompr_f>(sizeByte);
			base_t * out_pos = out_posColumn->get_data();
			
			int vec_count = merge_sorted_batch<t_vector_extension>::apply(
			  in_L_pos, in_R_pos,
			  out_pos,
			  in_L_dataCount, in_R_dataCount,
			  vectorState
			);
			
			out_posColumn->set_meta_data(vec_count, vec_count * sizeof(base_t));
			return out_posColumn;
		}
	};
	
}


#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_INTERSECT_UNCOMPR_H
