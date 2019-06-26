//
// Created by jpietrzyk on 28.05.19.
//

#ifndef MORPHSTORE_LINEAR_PROBING_H
#define MORPHSTORE_LINEAR_PROBING_H

#include <core/utils/preprocessor.h>

#include <vector/vector_primitives.h>
#include <vector/datastructures/hash_based/hash_utils.h>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>


namespace vector {

   /**
    * @brief Linear Probe Strategy for hash based data structures.
    * @details As every strategy for hash based data structures, different insert as well as lookup methods are provided.
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
      struct state_single_key_t {
            alignas(vector_size_byte::value) base_t m_KeyArray[vector_element_count::value];
            alignas(vector_size_byte::value) base_t m_ValueArray [vector_element_count::value] ;
            alignas(vector_size_byte::value) base_t m_IndexArray[vector_element_count::value];
            typename HashFunction<VectorExtension>::state_t m_HashState;
            typename index_aligner<VectorExtension>::template state_t<BiggestSupportedVectorExtension> m_AlignerState;
            typename index_resizer<VectorExtension, SPH>::state_t m_ResizerState;
            base_t * const m_KeyContainerStartPtr;
            base_t * const m_KeyContainerEndPtr;

         state_single_key_t(
            base_t * const p_KeyContainerStartPtr,
            size_t const p_ContainerBucketCount
         ) :
            m_ResizerState{p_ContainerBucketCount},
            m_KeyContainerStartPtr{p_KeyContainerStartPtr},
            m_KeyContainerEndPtr{p_KeyContainerStartPtr + m_ResizerState.m_ResizeValue - vector_element_count::value} { }
      };
      struct state_single_key_single_value_t {
         alignas(vector_size_byte::value) base_t m_KeyArray[vector_element_count::value];
         alignas(vector_size_byte::value) base_t m_ValueArray [vector_element_count::value] ;
         alignas(vector_size_byte::value) base_t m_IndexArray[vector_element_count::value];
         typename HashFunction<VectorExtension>::state_t m_HashState;
         typename index_aligner<VectorExtension>::template state_t<BiggestSupportedVectorExtension> m_AlignerState;
         typename index_resizer<VectorExtension, SPH>::state_t m_ResizerState;
         base_t * const m_KeyContainerStartPtr;
         base_t * const m_ValueContainerStartPtr;
         base_t * const m_KeyContainerEndPtr;

         state_single_key_single_value_t(
            base_t * const p_KeyContainerStartPtr,
            base_t * const p_ValueContainerStartPtr,
            size_t const p_ContainerBucketCount
         ) :
            m_ResizerState{p_ContainerBucketCount},
            m_KeyContainerStartPtr{p_KeyContainerStartPtr},
            m_ValueContainerStartPtr{p_ValueContainerStartPtr},
            m_KeyContainerEndPtr{p_KeyContainerStartPtr + m_ResizerState.m_ResizeValue - vector_element_count::value} { }
      };
      struct state_double_key_single_value_t {
         alignas(vector_size_byte::value) base_t m_FirstKeyArray[vector_element_count::value];
         alignas(vector_size_byte::value) base_t m_SecondKeyArray[vector_element_count::value];
         alignas(vector_size_byte::value) base_t m_ValueArray [vector_element_count::value] ;
         alignas(vector_size_byte::value) base_t m_IndexArray[vector_element_count::value];
         typename HashFunction<VectorExtension>::state_t m_HashState;
         typename index_aligner<VectorExtension>::template state_t<BiggestSupportedVectorExtension> m_AlignerState;
         typename index_resizer<VectorExtension, SPH>::state_t m_ResizerState;
         base_t * const m_FirstKeyContainerStartPtr;
         base_t * const m_SecondKeyContainerStartPtr;
         base_t * const m_ValueContainerStartPtr;
         base_t * const m_KeyContainerEndPtr;

         state_double_key_single_value_t(
            base_t * const p_FirstKeyContainerStartPtr,
            base_t * const p_SecondKeyContainerStartPtr,
            base_t * const p_ValueContainerStartPtr,
            size_t const p_ContainerBucketCount
         ) :
            m_ResizerState{p_ContainerBucketCount},
            m_FirstKeyContainerStartPtr{p_FirstKeyContainerStartPtr},
            m_SecondKeyContainerStartPtr{p_SecondKeyContainerStartPtr},
            m_ValueContainerStartPtr{p_ValueContainerStartPtr},
            m_KeyContainerEndPtr{p_FirstKeyContainerStartPtr + m_ResizerState.m_ResizeValue - vector_element_count::value} { }
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
      std::tuple<vector_t, vector_mask_t, uint8_t>
      lookup(
         vector_t const & p_InKeyVector,
         state_single_key_single_value_t & p_SearchState
      ) {
         vector_t const zeroVec = set1<VectorExtension, vector_base_t_granularity::value>(0);
         vector_mask_t resultMaskFound = 0;
         uint8_t resultCount = 0;
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
            base_t *currentSearchPtr = p_SearchState.m_KeyContainerStartPtr + index;
            keyVec = set1<VectorExtension, vector_base_t_granularity::value>(p_SearchState.m_KeyArray[pos] + 1 );
            bool done = false;
            while(!done) {
               vector_t loadedBucketsVec = load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
                  currentSearchPtr);
               searchOffset = equal<VectorExtension>::apply(loadedBucketsVec, keyVec);
               if(searchOffset != 0) {
                  p_SearchState.m_ValueArray[pos] =
                     p_SearchState.m_ValueContainerStartPtr[index + __builtin_ctz(searchOffset)];
                  resultMaskFound |= currentMask;
                  ++resultCount;
                  done = true;
               } else {
                  searchOffset = equal<VectorExtension>::apply(loadedBucketsVec, zeroVec);
                  if(searchOffset != 0) {
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_KeyContainerEndPtr)) {
                        currentSearchPtr += vector_element_count::value;
                        index += vector_element_count::value;
                     } else {
                        currentSearchPtr = p_SearchState.m_KeyContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
            currentMask = currentMask << 1;
         }
         return std::make_tuple(
            load<VectorExtension, iov::ALIGNED, vector_size_bit::value>( p_SearchState.m_ValueArray),
            resultMaskFound,
            resultCount
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
         state_single_key_t & p_SearchState
      ) {
         uint8_t resultCount = 0;
         vector_t const zeroVec = set1<VectorExtension, vector_base_t_granularity::value>(0);
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
            base_t *currentSearchPtr = p_SearchState.m_KeyContainerStartPtr + index;
            keyVec = set1<VectorExtension, vector_base_t_granularity::value>(p_SearchState.m_KeyArray[pos] + 1 );
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
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_KeyContainerEndPtr)) {
                        currentSearchPtr += vector_element_count::value;
                        index += vector_element_count::value;
                     } else {
                        currentSearchPtr = p_SearchState.m_KeyContainerStartPtr;
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
         state_single_key_t & p_SearchState
      ) {
         vector_t const zeroVec = set1<VectorExtension, vector_base_t_granularity::value>(0);
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
            base_t key = p_SearchState.m_KeyArray[pos] + 1;
            base_t *currentSearchPtr = p_SearchState.m_KeyContainerStartPtr + index;
            keyVec = set1<VectorExtension, vector_base_t_granularity::value>(key);
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
                     p_SearchState.m_KeyContainerStartPtr[ index + __builtin_ctz(searchOffset) ] = key;
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_KeyContainerEndPtr)) {
                        currentSearchPtr += vector_element_count::value;
                        index += vector_element_count::value;
                     } else {
                        currentSearchPtr = p_SearchState.m_KeyContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
         }
      }


      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      void
      insert(
         vector_t const & p_InKeyVector,
         vector_t const & p_InValueVector,
         state_single_key_single_value_t & p_SearchState
      ) {
         vector_t const zeroVec = set1<VectorExtension, vector_base_t_granularity::value>(0);
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
         store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
            p_SearchState.m_ValueArray,
            p_InValueVector
         );
         vector_t keyVec;
         vector_mask_t searchOffset;
         for(size_t pos = 0; pos < vector_element_count::value; ++pos) {
            base_t index = p_SearchState.m_IndexArray[pos];
            base_t key = p_SearchState.m_KeyArray[pos] + 1;
            base_t value = p_SearchState.m_ValueArray[pos];
            base_t *currentSearchPtr = p_SearchState.m_KeyContainerStartPtr + index;
            keyVec = set1<VectorExtension, vector_base_t_granularity::value>(key);
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
                     size_t targetIdx = index + __builtin_ctz(searchOffset);
                     p_SearchState.m_KeyContainerStartPtr[targetIdx] = key;
                     p_SearchState.m_ValueContainerStartPtr[targetIdx] = value;
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_KeyContainerEndPtr)) {
                        currentSearchPtr += vector_element_count::value;
                        index += vector_element_count::value;
                     } else {
                        currentSearchPtr = p_SearchState.m_KeyContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
         }
      }




      static
      std::tuple<
         vector_t,      // groupID vector register
         vector_t,      // groupExt vector register
         vector_mask_t, // active groupExt elements
         uint8_t        // Number of active groupExt elements
      >
      insert_and_lookup(
         vector_t const & p_InKeyVector,
         base_t & p_InStartPosFromKey,
         base_t & p_InStartValue,
         state_single_key_single_value_t & p_SearchState
      ) {
         vector_t const zeroVec = set1<VectorExtension, vector_base_t_granularity::value>(0);
         vector_mask_t activeGroupExtMask = 0;
         vector_mask_t currentMaskForGroupExtMask = 1;
         uint8_t activeGroupExtCount = 0;

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
            base_t key = p_SearchState.m_KeyArray[pos] + 1;

            base_t * currentSearchPtr = p_SearchState.m_KeyContainerStartPtr + index;
            keyVec = set1<VectorExtension, vector_base_t_granularity::value>(key);
            bool done = false;
            while(!done) {
               vector_t loadedBucketsVec = load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
                  currentSearchPtr);
               searchOffset = equal<VectorExtension>::apply(loadedBucketsVec, keyVec);
               if(searchOffset != 0) {
                  p_SearchState.m_ValueArray[ pos ] = p_SearchState.m_ValueContainerStartPtr[index + __builtin_ctz(searchOffset)];
                  done = true;
               } else {
                  searchOffset = equal<VectorExtension>::apply(loadedBucketsVec, zeroVec);
                  if(searchOffset != 0) {
                     size_t targetIdx = index + __builtin_ctz(searchOffset);
                     p_SearchState.m_KeyContainerStartPtr[targetIdx] = key;
                     p_SearchState.m_ValueContainerStartPtr[targetIdx] = p_InStartValue;
                     p_SearchState.m_ValueArray[ pos ] = p_InStartValue++;
                     p_SearchState.m_IndexArray[ pos ] = p_InStartPosFromKey;
                     activeGroupExtMask |= currentMaskForGroupExtMask;
                     ++activeGroupExtCount;
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_KeyContainerEndPtr)) {
                        currentSearchPtr += vector_element_count::value;
                        index += vector_element_count::value;
                     } else {
                        currentSearchPtr = p_SearchState.m_KeyContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
            currentMaskForGroupExtMask = currentMaskForGroupExtMask << 1;
            ++p_InStartPosFromKey;
         }
         return
            std::make_tuple(
               load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(p_SearchState.m_ValueArray),
               load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(p_SearchState.m_IndexArray),
               activeGroupExtMask,
               activeGroupExtCount
            );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      std::tuple<
         vector_t,      // groupID vector register
         vector_t,      // groupExt vector register
         vector_mask_t, // active groupExt elements
         uint8_t        // Number of active groupExt elements
      >
      insert_and_lookup(
         vector_t const & p_InKeysFirstVector,
         vector_t const & p_InKeySecondVector,
         base_t & p_InStartPosFromKey,
         base_t & p_InStartValue,
         state_double_key_single_value_t & p_SearchState
      ) {
         vector_t const zeroVec = set1<VectorExtension, vector_base_t_granularity::value>(0);
         vector_mask_t activeGroupExtMask = 0;
         vector_mask_t currentMaskForGroupExtMask = 1;
         uint8_t activeGroupExtCount = 0;

         store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
            p_SearchState.m_IndexArray,
            index_aligner<VectorExtension>::apply(
               index_resizer<VectorExtension, SPH>::apply(
                  HashFunction<VectorExtension>::apply(
                     p_InKeysFirstVector,
                     p_InKeySecondVector,
                     p_SearchState.m_HashState
                  ),
                  p_SearchState.m_ResizerState
               ),
               p_SearchState.m_AlignerState
            )
         );
         store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
            p_SearchState.m_FirstKeyArray,
            p_InKeysFirstVector
         );
         store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
            p_SearchState.m_SecondKeyArray,
            p_InKeySecondVector
         );
         vector_t keyFirstVec,keySecondVec;
         vector_mask_t searchOffset;

         for(size_t pos = 0; pos < vector_element_count::value; ++pos) {
            base_t index = p_SearchState.m_IndexArray[pos];
            base_t keyFirst = p_SearchState.m_FirstKeyArray[pos] + 1;
            base_t keySecond = p_SearchState.m_SecondKeyArray[pos];

            base_t * currentFirstKeySearchPtr = p_SearchState.m_FirstKeyContainerStartPtr + index;
            base_t * currentSecondKeySearchPtr = p_SearchState.m_SecondKeyContainerStartPtr + index;
            keyFirstVec = set1<VectorExtension, vector_base_t_granularity::value>(keyFirst);
            keySecondVec = set1<VectorExtension, vector_base_t_granularity::value>(keySecond);
            bool done = false;
            while(!done) {
               vector_t loadedFirstBucketsVec = load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
                  currentFirstKeySearchPtr);
               vector_t loadedSecondBucketsVec = load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
                  currentSecondKeySearchPtr);
               searchOffset =
                  (
                     equal<VectorExtension>::apply(loadedFirstBucketsVec, keyFirstVec)
                  &
                     equal<VectorExtension>::apply(loadedSecondBucketsVec, keySecondVec)
                  );
               if(searchOffset != 0) {
                  p_SearchState.m_ValueArray[ pos ] = p_SearchState.m_ValueContainerStartPtr[index + __builtin_ctz(searchOffset)];
                  done = true;
               } else {
                  searchOffset =
                     (
                        equal<VectorExtension>::apply(loadedFirstBucketsVec, zeroVec)
                     &
                        equal<VectorExtension>::apply(loadedSecondBucketsVec, zeroVec)
                     );
                  if(searchOffset != 0) {
                     size_t targetIdx = index + __builtin_ctz(searchOffset);
                     p_SearchState.m_FirstKeyContainerStartPtr[targetIdx] = keyFirst;
                     p_SearchState.m_SecondKeyContainerStartPtr[targetIdx] = keySecond;
                     p_SearchState.m_ValueContainerStartPtr[targetIdx] = p_InStartValue;
                     p_SearchState.m_ValueArray[pos] = p_InStartValue++;
                     p_SearchState.m_IndexArray[pos] = p_InStartPosFromKey;
                     activeGroupExtMask |= currentMaskForGroupExtMask;
                     ++activeGroupExtCount;
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentFirstKeySearchPtr < p_SearchState.m_KeyContainerEndPtr)) {
                        currentFirstKeySearchPtr += vector_element_count::value;
                        currentSecondKeySearchPtr += vector_element_count::value;
                        index += vector_element_count::value;
                     } else {
                        currentFirstKeySearchPtr = p_SearchState.m_FirstKeyContainerStartPtr;
                        currentSecondKeySearchPtr = p_SearchState.m_SecondKeyContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
            currentMaskForGroupExtMask = currentMaskForGroupExtMask << 1;
            ++p_InStartPosFromKey;
         }
         return
            std::make_tuple(
               load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(p_SearchState.m_ValueArray),
               load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(p_SearchState.m_IndexArray),
               activeGroupExtMask,
               activeGroupExtCount
            );
      }
   };

}
#endif //MORPHSTORE_LINEAR_PROBING_H
