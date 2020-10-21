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

#include <core/operators/interfaces/select.h>
#include <core/operators/uncompr/select.h>

#include <core/morphing/format.h>
#include <vector>


namespace morphstore {
	using namespace vectorlib;

	/// high level template
	template< class VectorExtension, template< class, int > class Operator >
	struct select_batch;
	
	template< class TVirtualVectorExtension, template< class, int > class Comparator >
	static void
	select_lambda(
	  const size_t begin, /// first vector
	  const size_t end, /// behind last vector
	  typename TVirtualVectorExtension::vector_helper_t::base_t in_predicate,
	  typename TVirtualVectorExtension::vector_helper_t::base_t const * in_dataPtr,
	  typename TVirtualVectorExtension::vector_helper_t::base_t *& out_dataPtr
	) {
		using VectorExtension = typename TVirtualVectorExtension::pps;
		IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
		
		size_t firstPos = begin * vector_element_count::value;
		
		/// predicate expanded to a whole vector register
		vector_t const predicateVector
		  = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(in_predicate);
		/// set constant incrementer
		vector_t const addVector
		  = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(vector_element_count::value);
		/// current positions for each element in the register
		vector_t positionVector
		  = vectorlib::set_sequence<VectorExtension, vector_base_t_granularity::value>(firstPos, 1);
		
		/// calc pointer of input according to offset
		base_t * in_ptr = ((base_t *) in_dataPtr) + (begin * vector_element_count::value);
		
		/// process each vector
		for (size_t i = begin; i < end; ++ i) {
			/// load data into vector register
			vector_t dataVector
			  = vectorlib::load
			    <VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>
			    (in_ptr);
			/// execute the filter for this vector
			vector_mask_t resultMask
			  = select_processing_unit<VectorExtension, Comparator>::apply(dataVector, predicateVector);
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
	
	
	template< class TVirtualVectorView, class TVectorExtension, template< class, int > class Comparator >
	struct select_batch<vv<TVirtualVectorView, TVectorExtension>, Comparator> {
		using TVE = vv<TVirtualVectorView, TVectorExtension>;
		IMPORT_VECTOR_BOILER_PLATE(TVE)
		
		MSV_CXX_ATTRIBUTE_FORCE_INLINE
		static void
		apply(
		  base_t const *& in_dataPtr,
		  base_t const    p_Predicate,
		  base_t       *& out_dataPtr,
		  size_t const    virtualVectorCnt,
		  int startid = 0
		) {
			
			/// calculate degree of parallelization: size of virtual vector / size of physical vector
			const uint16_t threadCnt
			  = std::max(vector_size_bit::value / TVectorExtension::vector_helper_t::size_bit::value, 1);
			uint64_t vectorCount
			  = virtualVectorCnt * vector_element_count::value / TVectorExtension::vector_helper_t::element_count::value;
			
			/// thread container
			std::thread * threads[threadCnt];
			
			/// step size / vector elements per pipeline
			size_t vectorsPerThread = vectorCount / threadCnt + ((vectorCount % threadCnt > 0) ? 1 : 0);
			
			size_t intermediate_elementCnt = vectorsPerThread * vector_element_count::value;
			
			column<uncompr_f> * intermediateColumns[threadCnt];
			base_t * intermediatePtr[threadCnt];
			base_t * intermediatePtrOrigin[threadCnt];
			

			
			/// init threads
			for(uint16_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
				size_t begin = threadIdx * vectorsPerThread;
				size_t end   = begin + vectorsPerThread;
				if(end > vectorCount)
					end = vectorCount;
				
				/// prepare intermediate column
				intermediateColumns[threadIdx]
				  = new column<uncompr_f>((end - begin) * vector_element_count::value * sizeof(base_t));
				intermediatePtr[threadIdx] = intermediateColumns[threadIdx]->get_data();
				
				threads[threadIdx] = new std::thread(
					/// lambda function
					select_lambda< TVE, Comparator >,
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
//				info("Combine Results");
				base_t * tmpPtr = intermediateColumns[threadIdx]->get_data();
				uint64_t dist = intermediatePtr[threadIdx] - tmpPtr;
//				info("Dist: " + std::to_string(dist));
				for(; tmpPtr < intermediatePtr[threadIdx]; ++tmpPtr){
					*out_dataPtr = *tmpPtr;
					++out_dataPtr;
				}
				delete intermediateColumns[threadIdx];
			}
			in_dataPtr += virtualVectorCnt * vector_element_count::value;
		}
	};
	
	
	template< class TVirtualVectorView, class TVectorExtension, template< class, int > class TComparator >
	struct select_t<vv<TVirtualVectorView, TVectorExtension>, TComparator, uncompr_f, uncompr_f> {
		using TVE = vv<TVirtualVectorView, TVectorExtension>;
		IMPORT_VECTOR_BOILER_PLATE(TVE)
		
		MSV_CXX_ATTRIBUTE_FORCE_INLINE
		static column <uncompr_f> const *
		apply(
		  column <uncompr_f> const * const in_dataColumn,
		  base_t const in_predicate,
		  const size_t out_posCountEstimate = 0
		) {
			/// fetch metadata & data pointer
			size_t const in_dataCount = in_dataColumn->get_count_values();
			base_t const * in_dataPtr = in_dataColumn->get_data();
			
			/// calc variables
			size_t const sizeByte =
              bool(out_posCountEstimate) ? (out_posCountEstimate * sizeof(base_t)) : in_dataColumn->get_size_used_byte();
			size_t const vectorCount    = in_dataCount / vector_element_count::value;
			size_t const remainderCount = in_dataCount % vector_element_count::value;
			
			/// create output column
			auto out_dataCol = new column<uncompr_f>(sizeByte);
			base_t * out_dataPtr = out_dataCol->get_data();
			base_t * const out_dataPtrOrigin = const_cast< base_t * const >(out_dataPtr);
			
			
			/// process data vectorized
			select_batch<TVE, TComparator>::apply(in_dataPtr, in_predicate, out_dataPtr, vectorCount);
			
//			info("RemainderCount: " + std::to_string(remainderCount));
			/// process remaining data scalar
			select_batch<scalar<v64<uint64_t>>, TComparator>::apply(
              in_dataPtr, in_predicate, out_dataPtr, remainderCount, vectorCount * vector_element_count::value
			);
			/// set metadata for output column
			size_t const outDataCount = out_dataPtr - out_dataPtrOrigin;
			out_dataCol->set_meta_data(outDataCount, outDataCount * sizeof(base_t));
			
			return out_dataCol;
		}
	};
	
} /// namespace morphstore


#endif //MORPHSTORE_SELECT_UNCOMPR_H
