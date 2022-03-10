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


#ifndef MORPHSTORE_PROJECT_UNCOMPR_H
#define MORPHSTORE_PROJECT_UNCOMPR_H

#include <core/operators/interfaces/project.h>
#include <core/operators/uncompr/project.h>

#include <core/utils/preprocessor.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <thread>

namespace morphstore {
    using namespace vectorlib;
    
    /// Declarations @todo move into interface?
    template< class t_vector_extension >
    struct project_batch;
    
    template< class t_vector_extension >
    struct project_core;
    
//    template< class t_vector_extension >
//    struct project_processing_unit;
//
//
//    template< class TVirtualVectorView, class TVectorExtension >
//    struct project_processing_unit<vv<TVirtualVectorView, TVectorExtension>> {
//        using t_vector_extension = vv<TVirtualVectorView, TVectorExtension>;
//        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
//
//        MSV_CXX_ATTRIBUTE_FORCE_INLINE
//        static vector_t
//        apply(
//          base_t const * p_DataPtr,
//          vector_t p_PosVector
//        ) {
//            //@todo: Is it better to avoid gather here?
//            return
//              vectorlib::gather<t_vector_extension, vector_size_bit::value, sizeof(base_t)>(p_DataPtr, p_PosVector);
//        }
//    };
    
    
    template< class TVirtualVectorView, class TVectorExtension >
    struct project_core<vv_old<TVirtualVectorView, TVectorExtension>> {
        using t_vector_extension = TVectorExtension;
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static void
        lambda(
          const size_t begin,
          const size_t end,
          base_t const * in_dataPtr,
          base_t const * in_posPtr,
          base_t * out_dataPtr
        ){
            /// execute primitives for each physical vector
            for(size_t physical_vector_id = begin; physical_vector_id < end; ++physical_vector_id){
                /// load positions into vector register using physical vector extension (i.e. avx512, avx2, sse ...)
                vector_t posVector
                  = vectorlib::load
                    <t_vector_extension, vectorlib::iov::ALIGNED, vector_size_bit::value>
                    (in_posPtr + physical_vector_id * vector_element_count::value);
                /// lookup data using positional information in posVector
                vector_t lookupVector
                  = project_processing_unit<t_vector_extension>::apply(in_dataPtr, posVector);
                /// store resulting vector register in output column
                vectorlib::store
                  <t_vector_extension, vectorlib::iov::ALIGNED, vector_size_bit::value>
                  (out_dataPtr, lookupVector);
                
                out_dataPtr += vector_element_count::value;
            }
        }
        
    };
    
    
    template< class TVirtualVectorView, class TVectorExtension >
    struct project_batch<vv_old<TVirtualVectorView, TVectorExtension>> {
        using t_vector_extension = vv_old<TVirtualVectorView, TVectorExtension>;
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static void
        apply(
          base_t const *& in_dataPtr,
          base_t const *& in_posPtr,
          base_t *& out_dataPtr,
          size_t const virtualVectorCount
        ) {
			/// calculate degree of parallelization: size of virtual vector / size of physical vector
			/// minimum = 1
			const uint16_t threadCnt
			  = std::max(vector_size_bit::value / TVectorExtension::vector_helper_t::size_bit::value, 1);
			uint64_t vectorCount
			  = virtualVectorCount * vector_element_count::value / TVectorExtension::vector_helper_t::element_count::value;
     
			/// thread container
			std::thread * threads[threadCnt];
//			base_t * intermediates[threadCnt];
   
			/// step size // vector elements per pipeline
			size_t vectorsPerThread = vectorCount / threadCnt + ((vectorCount % threadCnt > 0) ? 1 : 0);
   
			/// create output array for each thread
//			for(uint16_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
//			    intermediates[threadIdx] = new base_t[vectorsPerThread * TVectorExtension::vector_helper_t::element_count::value];
//			}
			
			/// init threads
			for(uint16_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
				size_t begin = threadIdx * vectorsPerThread;
				size_t end = (threadIdx + 1) * vectorsPerThread;
				if(end > vectorCount)
					end = vectorCount;

				threads[threadIdx] = new std::thread(
					/// lambda function
					project_core<t_vector_extension>::lambda,
					/// parameters
					begin, end, in_dataPtr, in_posPtr, out_dataPtr
				);
				out_dataPtr += (end - begin) * TVectorExtension::vector_helper_t::element_count::value;
			}
			
			/// wait for threads to finish
			for(uint16_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
				threads[threadIdx]->join();
				delete threads[threadIdx];
			}
			
			/// combine results
			/// ...
        }
    };
    
    
    template< class TVirtualVectorView, class TVectorExtension >
    struct project_t<vv_old<TVirtualVectorView, TVectorExtension>, uncompr_f, uncompr_f, uncompr_f> {
        using TVE = vv_old<TVirtualVectorView, TVectorExtension>;
        IMPORT_VECTOR_BOILER_PLATE(TVE)
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static column <uncompr_f> const *
        apply(
          column <uncompr_f> const * const in_dataColumn,
          column <uncompr_f> const * const in_posColumn
        ) {
            /// fetch metadata & data pointer
            size_t const in_posCount = in_posColumn->get_count_values();
            size_t const inUsedBytes = in_posColumn->get_size_used_byte();
            base_t const * inDataPtr = in_dataColumn->get_data();
            base_t const * inPosPtr = in_posColumn->get_data();
            
            /// calc variables
            size_t const vectorCount = in_posCount / vector_element_count::value;
            size_t const remainderCount = in_posCount % vector_element_count::value;
            
            /// create output column
            auto outDataCol = new column<uncompr_f>(inUsedBytes);
            base_t * outDataPtr = outDataCol->get_data();
            
            /// process data vectorized
            project_batch<TVE>::apply(inDataPtr, inPosPtr, outDataPtr, vectorCount);
            
            /// process remaining data scalar
            project_batch<scalar<v64<uint64_t>>>::apply(inDataPtr, inPosPtr, outDataPtr, remainderCount);
            
            outDataCol->set_meta_data(in_posCount, inUsedBytes);
            
            return outDataCol;
        }
    };
    
} /// namespace morphstore



#endif //MORPHSTORE_PROJECT_UNCOMPR_H
