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


#ifndef MORPHSTORE_CORE_OPERATORS_VIRTUAL_VECTORIZED_AGG_SUM_ALL_H
#define MORPHSTORE_CORE_OPERATORS_VIRTUAL_VECTORIZED_AGG_SUM_ALL_H


#include <core/operators/interfaces/agg_sum_all.h>
#include <core/operators/uncompr/agg_sum_all.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <type_traits>
#include <thread>
#include <iostream>


namespace morphstore {

   using namespace vectorlib;
   
	
    template<class TVirtualVectorView, class TVectorExtension>
	struct agg_sum_processing_unit<vv<TVirtualVectorView, TVectorExtension>> {
   	    using TVVE = vv<TVirtualVectorView, TVectorExtension>;
		IMPORT_VECTOR_BOILER_PLATE(TVVE)
		
		struct state_t {
			typename TVectorExtension::vector_t resultVec;
			
			state_t(void): resultVec( vectorlib::set1<TVectorExtension, TVectorExtension::vector_helper_t::granularity::value>(0 ) ) { }
			state_t(base_t p_Data): resultVec(vectorlib::set1<scalar<v64<uint64_t>>,64>(p_Data)){}
		};
		
		MSV_CXX_ATTRIBUTE_FORCE_INLINE
		static void
		apply(typename TVectorExtension::vector_t const & p_DataVector, state_t & p_State) {
			p_State.resultVec = vectorlib::add<TVectorExtension, TVectorExtension::vector_helper_t::granularity::value>::apply(
				p_State.resultVec, p_DataVector
			);
		}
		
		MSV_CXX_ATTRIBUTE_FORCE_INLINE
		static base_t
		finalize(typename agg_sum_processing_unit<TVVE>::state_t & p_State) {
			return vectorlib::hadd<TVectorExtension, TVectorExtension::vector_helper_t::granularity::value>::apply(p_State.resultVec );
		}
   };

    template<class TVirtualVectorView, class TVectorExtension>
	struct agg_sum_batch<vv<TVirtualVectorView, TVectorExtension>> {
   	    using TVVE = vv<TVirtualVectorView, TVectorExtension>;
		IMPORT_VECTOR_BOILER_PLATE(TVVE)
		
		MSV_CXX_ATTRIBUTE_FORCE_INLINE
		static base_t
		apply(
		  base_t const *& in_dataPtr,
		  size_t const virtualVectorCnt,
		  MSV_CXX_ATTRIBUTE_PPUNUSED
		  typename agg_sum_processing_unit<TVVE>::state_t &p_State
		) {
			/// shortcut state type of physical processing style
			using state_t = typename agg_sum_processing_unit<TVectorExtension>::state_t;
			
			/// calculate degree of parallelization: size of virtual vector / size of physical vector
			/// minimum = 1
			const uint16_t threadCnt
			  = std::max(vector_size_bit::value / TVectorExtension::vector_helper_t::size_bit::value, 1);
			uint64_t vectorCount
			  = virtualVectorCnt * vector_element_count::value / TVectorExtension::vector_helper_t::element_count::value;
			
//			std::cout << "Thread count: " << threadCnt << std::endl;
//			std::cout << "virtual vector count: " << virtualVectorCnt << std::endl;
			
			/// thread container
			std::thread * threads[threadCnt];
			
			/// states // one for each pipeline
			state_t p_States[threadCnt];
			
			/// step size // vector elements per pipeline
			size_t vectorsPerThread = vectorCount / threadCnt + ((vectorCount % threadCnt > 0) ? 1 : 0);
			
			/// This function is used to parallelize the load and processing step of this operator.
			/// To ensure disjoint input data for each thread a range (offset + end) is hand over.
			auto lambda =
				[] (const size_t begin, const size_t end, state_t * p_States, const base_t * p_DataPtr, const uint16_t pipe/* */) {
					for(size_t i = begin; i < end; ++i) {
						typename TVectorExtension::vector_t dataVector =
							vectorlib::load<TVectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>
								( p_DataPtr + (i * TVectorExtension::vector_helper_t::element_count::value) );
						agg_sum_processing_unit<TVectorExtension>::apply(dataVector, p_States[pipe]);
					}
				};
			
			/// init threads
			for(uint16_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
				size_t begin = threadIdx * vectorsPerThread;
				size_t end = (threadIdx + 1) * vectorsPerThread;
				if(end > vectorCount)
					end = vectorCount;

				threads[threadIdx] = new std::thread(
					/// lambda function
					lambda,
					/// parameters
					begin, end, p_States, in_dataPtr, threadIdx
				);
			}

			/// wait for threads to finish
			for(uint16_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
				threads[threadIdx]->join();
				delete threads[threadIdx];
			}

			/// aggregate results of all pipelines
			base_t result = agg_sum_processing_unit<TVectorExtension>::finalize(p_States[0]);
			for(uint16_t threadIdx = 1; threadIdx < threadCnt; ++threadIdx){
				result += agg_sum_processing_unit<TVectorExtension>::finalize(p_States[threadIdx]);
			}
			
			in_dataPtr += virtualVectorCnt * vector_element_count::value;
			
			return result;
		}
	};

//	template<class TVirtualVectorView, class TVectorExtension>
//	struct agg_sum_all_t<vv<TVirtualVectorView, TVectorExtension>, uncompr_f, uncompr_f> {
//   	    using TVE = vv<TVirtualVectorView, TVectorExtension>;
//		IMPORT_VECTOR_BOILER_PLATE( TVE )
//
//		MSV_CXX_ATTRIBUTE_FORCE_INLINE
//		static const column<uncompr_f> *
//		apply( column< uncompr_f > const * const in_dataColumn ) {
//		    typename agg_sum_processing_unit<TVE>::state_t vectorState;
//
////		    size_t const vectorCount = in_dataColumn->get_count_values() / TVectorExtension::vector_helper_t::element_count::value;
//		    size_t const vectorCount = in_dataColumn->get_count_values() / vector_element_count::value;
////		    size_t const remainderCount = in_dataColumn->get_count_values() % TVectorExtension::vector_helper_t::element_count::value;
//		    size_t const remainderCount = in_dataColumn->get_count_values() % vector_element_count::value;
//		    base_t const * dataPtr = in_dataColumn->get_data();
//
//		    base_t t = agg_sum_batch<TVE>::apply( dataPtr, vectorCount, vectorState );
//		    typename agg_sum_processing_unit<scalar<v64<uint64_t>>>::state_t scalarState( t );
//
//
//		    /// arg_sum_batch for vv<...> does not increment this pointer, so we have to do this manually
////		    dataPtr += vectorCount * TVectorExtension::vector_helper_t::element_count::value;
//
//            std::cout << "Remainder Count: " << remainderCount << std::endl;
//
//			base_t result = agg_sum_batch<scalar < v64 < uint64_t > >>::apply( dataPtr, remainderCount, scalarState );
//
//		    auto outDataCol = new column<uncompr_f>(sizeof(base_t));
//		    base_t * const outData = outDataCol->get_data();
//		    *outData = result;
//		    outDataCol->set_meta_data(1, sizeof(base_t));
//
//		    return outDataCol;
//		}
//    };
   
 
} /// namespace morphstore



#endif // MORPHSTORE_CORE_OPERATORS_VIRTUAL_VECTORIZED_AGG_SUM_ALL_H