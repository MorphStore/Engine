//
// Created by jpietrzyk on 28.05.19.
//

#ifndef MORPHSTORE_LINEAR_PROBING_H
#define MORPHSTORE_LINEAR_PROBING_H
#include <vector/general_vector.h>
#include <core/utils/preprocessor.h>
#include <vector/primitives/create.h>
#include <vector/primitives/io.h>
#include <vector/primitives/logic.h>
#include <vector/datastructures/hash_based/hash_utils.h>

namespace vector {

   /**
    * @brief Linear Probe Strategy for hash based data structures.
    * @details @todo
    * @tparam VectorExtension Vector extension which is used for probing.
    * @tparam BiggestSupportedVectorExtension Biggest vector extension the linear search should be able to work with.
    * @tparam HashFunction Struct which provides an static apply function to hash a vector register (VectorExtension::vector_t).
    * @tparam SPH Size policy which is needed for vector::index_resizer
    * (either size_policy_hash::ARBITRARY or size_policy_hash::EXPONENTIAL).
    */
   template<
      class VectorExtension,
      class BiggestSupportedVectorExtension,
      template<class> class HashFunction,
      size_policy_hash SPH
   >
   struct scalar_key_vectorized_linear_search {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         MSV_CXX_ATTRIBUTE_HOT
            alignas(vector_size_byte::value) base_t m_KeyArray[vector_element_count::value];
         MSV_CXX_ATTRIBUTE_HOT
            alignas(vector_size_byte::value) base_t m_IndexArray[vector_element_count::value];
         MSV_CXX_ATTRIBUTE_HOT
            typename HashFunction<VectorExtension>::state_t m_HashState;
         MSV_CXX_ATTRIBUTE_HOT
            typename index_aligner<VectorExtension>::template state_t<BiggestSupportedVectorExtension> m_AlignerState;
         MSV_CXX_ATTRIBUTE_HOT
            typename index_resizer<VectorExtension, SPH>::state_t m_ResizerState;
         MSV_CXX_ATTRIBUTE_HOT
            base_t * const m_ContainerStartPtr;
         MSV_CXX_ATTRIBUTE_HOT
            base_t * const m_ContainerEndPtr;

         state_t(
            base_t * const p_ContainerStartPtr,
            size_t const p_ContainerBucketCount
         ) :
            m_ResizerState{p_ContainerBucketCount},
            m_ContainerStartPtr{p_ContainerStartPtr},
            m_ContainerEndPtr{p_ContainerStartPtr + m_ResizerState.m_ResizeValue - vector_element_count::value} { }
      };
      /**
       *
       * @param p_InKeyVector
       * @param p_SearchState
       * @return Tuple containing a vector register with the indices where either the corresponding key matched, or an
       * empty bucket was found, a vector register mask with a bit set to one if the corresponding bucket matched and a
       * vector register mask with a bit set to one if the corresponding bucket is empty.
       */
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      std::tuple<vector_t, vector_mask_t>
      lookup_pos(
         vector_t const & p_InKeyVector,
         state_t & p_SearchState
      ) {
         vector_t const zeroVec = set1<VectorExtension>(0);
         vector_mask_t resultMaskFound = 0;
         store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
            p_SearchState.m_IndexArray,
            index_aligner<VectorExtension>::apply(
               index_resizer<VectorExtension, SPH>::apply(
                  HashFunction<VectorExtension>::apply(
                     p_InKeyVector,
                     p_SearchState.m_HashState
                  ),
                  p_SearchState.m_ResizerState
               ),
               p_SearchState.m_AlignerState
            )
         );
         store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
            p_SearchState.m_KeyArray,
            p_InKeyVector
         );
         vector_t keyVec;
         vector_mask_t searchOffset;
         vector_mask_t currentMask = 1;
         for(size_t pos = 0; pos < vector_element_count::value; ++pos) {
            base_t index = p_SearchState.m_IndexArray[pos];
            base_t *currentSearchPtr = p_SearchState.m_ContainerStartPtr + index;
            keyVec = set1<VectorExtension>(p_SearchState.m_KeyArray[pos]);
            searchOffset = 0;
            bool done = false;
            while(!done) {
               vector_t loadedBucketsVec = load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
                  currentSearchPtr);
               searchOffset = equal<VectorExtension>::apply(loadedBucketsVec, keyVec);
               if(searchOffset != 0) {
                  p_SearchState.m_IndexArray[pos] = index + __builtin_ctz(searchOffset);
                  resultMaskFound |= currentMask;
                  done = true;
               } else {
                  searchOffset = equal<VectorExtension>::apply(loadedBucketsVec, zeroVec);
                  if(searchOffset != 0) {
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_ContainerEndPtr)) {
                        currentSearchPtr += vector_element_count::value;
                        index += vector_element_count::value;
                     } else {
                        currentSearchPtr = p_SearchState.m_ContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
            currentMask = currentMask << 1;
         }
         return std::make_tuple(
            load<VectorExtension, iov::ALIGNED, vector_size_bit::value>( p_SearchState.m_IndexArray),
            resultMaskFound
         );
      }

      /**
       *
       * @param p_InKeyVector
       * @param p_SearchState
       * @return Tuple containing a vector register with the indices where either the corresponding key matched, or an
       * empty bucket was found, a vector register mask with a bit set to one if the corresponding bucket matched and a
       * vector register mask with a bit set to one if the corresponding bucket is empty.
       */
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      std::pair< vector_mask_t, uint8_t >
      lookup(
         vector_t const & p_InKeyVector,
         state_t & p_SearchState
      ) {
         uint8_t resultCount = 0;
         vector_t const zeroVec = set1<VectorExtension>(0);
         vector_mask_t resultMaskFound = 0;
         store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
            p_SearchState.m_IndexArray,
            index_aligner<VectorExtension>::apply(
               index_resizer<VectorExtension, SPH>::apply(
                  HashFunction<VectorExtension>::apply(
                     p_InKeyVector,
                     p_SearchState.m_HashState
                  ),
                  p_SearchState.m_ResizerState
               ),
               p_SearchState.m_AlignerState
            )
         );
         store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
            p_SearchState.m_KeyArray,
            p_InKeyVector
         );
         vector_t keyVec;
         vector_mask_t searchOffset;
         vector_mask_t currentMask = 1;
         for(size_t pos = 0; pos < vector_element_count::value; ++pos) {
            base_t index = p_SearchState.m_IndexArray[pos];
            base_t *currentSearchPtr = p_SearchState.m_ContainerStartPtr + index;
            keyVec = set1<VectorExtension>(p_SearchState.m_KeyArray[pos]);
            searchOffset = 0;
            bool done = false;
            while(!done) {
               vector_t loadedBucketsVec = load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
                  currentSearchPtr);
               searchOffset = equal<VectorExtension>::apply(loadedBucketsVec, keyVec);
               if(searchOffset != 0) {
                  resultMaskFound |= currentMask;
                  ++resultCount;
                  done = true;
               } else {
                  searchOffset = equal<VectorExtension>::apply(loadedBucketsVec, zeroVec);
                  if(searchOffset != 0) {
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_ContainerEndPtr)) {
                        currentSearchPtr += vector_element_count::value;
                        index += vector_element_count::value;
                     } else {
                        currentSearchPtr = p_SearchState.m_ContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
            currentMask = currentMask << 1;
         }
         return std::make_pair(resultMaskFound, resultCount);
      }


      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      void
      insert(
         vector_t const & p_InKeyVector,
         state_t & p_SearchState
      ) {
         vector_t const zeroVec = set1<VectorExtension>(0);
         vector_mask_t resultMaskFound = 0;
         store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
            p_SearchState.m_IndexArray,
            index_aligner<VectorExtension>::apply(
               index_resizer<VectorExtension, SPH>::apply(
                  HashFunction<VectorExtension>::apply(
                     p_InKeyVector,
                     p_SearchState.m_HashState
                  ),
                  p_SearchState.m_ResizerState
               ),
               p_SearchState.m_AlignerState
            )
         );
         store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
            p_SearchState.m_KeyArray,
            p_InKeyVector
         );
         vector_t keyVec;
         vector_mask_t searchOffset;
         for(size_t pos = 0; pos < vector_element_count::value; ++pos) {
            base_t index = p_SearchState.m_IndexArray[pos];
            base_t key = p_SearchState.m_KeyArray[pos];
            base_t *currentSearchPtr = p_SearchState.m_ContainerStartPtr + index;
            keyVec = set1<VectorExtension>(key);
            searchOffset = 0;
            bool done = false;
            while(!done) {
               vector_t loadedBucketsVec = load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
                  currentSearchPtr);
               searchOffset = equal<VectorExtension>::apply(loadedBucketsVec, keyVec);
               if(searchOffset != 0) {
                  done = true;
               } else {
                  searchOffset = equal<VectorExtension>::apply(loadedBucketsVec, zeroVec);
                  if(searchOffset != 0) {
                     p_SearchState.m_ContainerStartPtr[ index + __builtin_ctz(searchOffset) ] = key;
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_ContainerEndPtr)) {
                        currentSearchPtr += vector_element_count::value;
                        index += vector_element_count::value;
                     } else {
                        currentSearchPtr = p_SearchState.m_ContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
         }
      }

   };


}
#endif //MORPHSTORE_LINEAR_PROBING_H
