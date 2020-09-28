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


#ifndef MORPHSTORE_SELECT_UNCOMPR_H
#define MORPHSTORE_SELECT_UNCOMPR_H

#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <thread>

/// don't include this!! this code does not implement the correct interface
//#include <core/operators/interfaces/select.h>

#include <core/operators/general_vectorized/select_uncompr.h>
#include <core/morphing/format.h>
#include <vector>


namespace morphstore {
	using namespace vectorlib;

	/// high level template
	template< class VectorExtension, template< class, int > class Operator >
	struct select_batch;
	
//	template< class VectorExtension, template< class, int > class Operator >
//	struct select_t;
	
	template< class TVirtualVectorExtension, template< class, int > class Operator >
	static void select_lambda(
	  const size_t begin, /// first vector
	  const size_t end, /// behind last vector
	  typename TVirtualVectorExtension::vector_helper_t::base_t p_Predicate,
	  typename TVirtualVectorExtension::vector_helper_t::base_t const * in_dataPtr,
	  typename TVirtualVectorExtension::vector_helper_t::base_t *& out_dataPtr
	) {
		using VectorExtension = typename TVirtualVectorExtension::pps;
		IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
		
		size_t firstPos = begin * vector_element_count::value;
		
		/// predicate expanded to a whole vector register
		vector_t const predicateVector
		  = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(p_Predicate);
		/// incrementer
		vector_t const addVector
		  = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(vector_element_count::value);
		/// current positions for each element in the register
		vector_t positionVector
		  = vectorlib::set_sequence<VectorExtension, vector_base_t_granularity::value>(firstPos, 1);
		
		base_t * in_ptr = ((base_t *) in_dataPtr) + (begin * vector_element_count::value);
		for (size_t i = begin; i < end; ++ i) {
			/// load vector register
			vector_t dataVector
			  = vectorlib::load
			    <VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>
			    (in_ptr);
			/// execute select
			vector_mask_t resultMask
			  = select_processing_unit<VectorExtension, Operator>::apply(dataVector, predicateVector);
			/// store positions of selected values
			vectorlib::compressstore
			  <VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>
			  (out_dataPtr, positionVector, resultMask);
			
			/// increment position vector
			positionVector
			  = vectorlib::add<VectorExtension, vector_base_t_granularity::value>::apply(positionVector, addVector);
			/// increment pointer of incoming data by element count of the vector register
			in_ptr += vector_element_count::value;
			/// increment pointer of outgoing data by match count
			out_dataPtr += vectorlib::count_matches<VectorExtension>::apply(resultMask);
		}
	}
	
	
	template< class TVirtualVectorView, class TVectorExtension, template< class, int > class Operator >
	struct select_batch<vv<TVirtualVectorView, TVectorExtension>, Operator> {
		using TVE = vv<TVirtualVectorView, TVectorExtension>;
		IMPORT_VECTOR_BOILER_PLATE(TVE)
		
		MSV_CXX_ATTRIBUTE_FORCE_INLINE
		static void apply(
		  base_t const *& in_dataPtr,
		  base_t const    p_Predicate,
		  base_t       *& out_dataPtr,
		  size_t const    virtualVectorCnt,
		  int startid = 0
		) {
			
			/// calculate degree of parallelization: size of virtual vector / size of physical vector
			const uint16_t threadCnt
			  = std::max(vector_size_bit::value / TVectorExtension::vector_helper_t::size_bit::value, 1);
			
			/// thread container
			std::thread * threads[threadCnt];
			
			/// step size / vector elements per pipeline
			size_t vectorsPerThread = virtualVectorCnt / threadCnt + ((virtualVectorCnt % threadCnt > 0) ? 1 : 0);
			
			size_t intermediate_elementCnt = vectorsPerThread * vector_element_count::value;
			
			column<uncompr_f> * intermediateColumns[threadCnt];
			base_t * intermediatePtr[threadCnt];
			base_t * intermediatePtrOrigin[threadCnt];
			

			
			/// init threads
			for(uint16_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
				size_t begin = threadIdx * vectorsPerThread;
				size_t end   = begin + vectorsPerThread;
				if(end > virtualVectorCnt)
					end = virtualVectorCnt;
				
				/// prepare intermediate column
				intermediateColumns[threadIdx]
				  = new column<uncompr_f>((end - begin) * vector_element_count::value * sizeof(base_t));
				intermediatePtr[threadIdx] = intermediateColumns[threadIdx]->get_data();
				
				threads[threadIdx] = new std::thread(
					/// lambda function
					select_lambda< TVE, Operator >,
					/// parameters
					begin, end, p_Predicate, std::ref(in_dataPtr), std::ref(intermediatePtr[threadIdx])/**/
				);
			}
			
			/// wait for threads to finish
			for(uint16_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
				threads[threadIdx]->join();
				delete threads[threadIdx];
			}
			
			/// combine results
			for(uint16_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
				info("Combine Results");
				base_t * tmpPtr = intermediateColumns[threadIdx]->get_data();
				uint64_t dist = intermediatePtr[threadIdx] - tmpPtr;
				info("Dist: " + std::to_string(dist));
				for(; tmpPtr < intermediatePtr[threadIdx]; ++tmpPtr){
					*out_dataPtr = *tmpPtr;
					++out_dataPtr;
				}
			}
		}
	};
	
	
	/// @todo : adapt to interface (adopted (wrong) format from uncompressed select operator to stay consistent ... for now)
	template< class TVirtualVectorView, class TVectorExtension, template< class, int > class Operator >
	struct select_t<vv<TVirtualVectorView, TVectorExtension>, Operator> {
		using TVE = vv<TVirtualVectorView, TVectorExtension>;
		IMPORT_VECTOR_BOILER_PLATE(TVE)
		
		MSV_CXX_ATTRIBUTE_FORCE_INLINE
		static column <uncompr_f> const * apply(
		  column <uncompr_f> const * const p_DataColumn,
		  base_t const p_Predicate,
		  const size_t outPosCountEstimate = 0
		) {
			/// fetch metadata
			size_t const inDataCount = p_DataColumn->get_count_values();
			base_t const * inDataPtr = p_DataColumn->get_data();
			size_t const sizeByte =
			  bool(outPosCountEstimate) ? (outPosCountEstimate * sizeof(base_t)) : p_DataColumn->get_size_used_byte();
			
			/// create output column
			auto outDataCol = new column<uncompr_f>(sizeByte);
			base_t * outDataPtr = outDataCol->get_data();
			base_t * const outDataPtrOrigin = const_cast< base_t * const >(outDataPtr);
			
			size_t const vectorCount    = inDataCount / vector_element_count::value;
			size_t const remainderCount = inDataCount % vector_element_count::value;
			
			/// process data vectorized
			select_batch<TVE, Operator>::apply(inDataPtr, p_Predicate, outDataPtr, vectorCount);
			//... increase data ptr
			
			info("RemainderCount: " + std::to_string(remainderCount));
			/// process remaining data scalar
			select_batch<scalar<v64<uint64_t>>, Operator>::apply(
			  inDataPtr, p_Predicate, outDataPtr, remainderCount, vectorCount * vector_element_count::value
			);
			/// set metadata for output column
			size_t const outDataCount = outDataPtr - outDataPtrOrigin;
			outDataCol->set_meta_data(outDataCount, outDataCount * sizeof(base_t));
			
			return outDataCol;
		}
	};
	
} // namespace morphstore


#endif //MORPHSTORE_SELECT_UNCOMPR_H
