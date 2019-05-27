//
// Created by jpietrzyk on 21.05.19.
//

#ifndef MORPHSTORE_SET_STRATEGIES_H
#define MORPHSTORE_SET_STRATEGIES_H

#include <vector/general_vector.h>
#include <core/utils/preprocessor.h>
#include <vector/datastructures/set_utils.h>
#include <vector/primitives/compare.h>
#include <vector/primitives/io.h>
#include <vector/primitives/create.h>

namespace vector {

   template<
      class VectorExtension,
      class BiggestSupportedVectorExtension,
      template<class> class HashFunction,
      size_policy_set SPS
   >
   struct scalar_key_vectorized_linear_search {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         base_t * const m_Start;
         base_t * const m_End;
         state_t(
            base_t * const p_Start,
            typename key_resizer<VectorExtension, SPS>::state_t const & p_State
         ) :
            m_Start{ p_Start },
            m_End{ p_Start + p_State.m_ResizeValue - vector_element_count::value} {}
      };

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      void
      insert_processing_unit(
         base_t const p_Key,
         base_t * const p_PtrToInsert
      ) {
         *(p_PtrToInsert) = p_Key;
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      bool
      lookup_processing_unit(
         base_t const p_Key,
         base_t *& p_Position,
         state_t const & p_SearcherState
      ) {
         vector_t const zeroVec = set1<VectorExtension>(0);
         vector_t const keyVec = set1<VectorExtension>(p_Key);
         vector_mask_t searchOffset = 0;
         base_t * current;
         base_t * next = p_Position;
         do{
            current = next;
            next = (( current < p_SearcherState.m_End ) ? current + vector_element_count::value : p_SearcherState.m_Start);
            vector_t searchVec = load<VectorExtension, iov::ALIGNED, vector_size_bit::value>( current );
            if( MSV_CXX_ATTRIBUTE_UNLIKELY(equal<VectorExtension>::apply( searchVec, keyVec ) != 0 )) {
               //p_Position = current + __builtin_ctz( searchOffset );
               return true;
            }
            searchOffset = equal<VectorExtension>::apply( searchVec, zeroVec );
         }while(searchOffset == 0 );
         p_Position = current + __builtin_ctz( searchOffset );
         return false;
      }



      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      void
      build_batch(
         base_t *& p_KeyPtr,
         base_t * const p_Data,
         size_t const p_Count,
         typename key_resizer<VectorExtension, SPS>::state_t const & p_ResizerState
      ) {
         MSV_CXX_ATTRIBUTE_HOT alignas(vector_size_byte::value) base_t hashArray[vector_element_count::value];
         MSV_CXX_ATTRIBUTE_HOT typename HashFunction<VectorExtension>::state_t hashState;
         MSV_CXX_ATTRIBUTE_HOT typename key_aligner<VectorExtension>::template state_t<BiggestSupportedVectorExtension> alignerState;
         state_t searcherState(p_Data, p_ResizerState);

         for(size_t i = 0; i < p_Count; ++i) {
            store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
               hashArray,
               key_aligner<VectorExtension>::apply(
                  key_resizer<VectorExtension, SPS>::apply(
                     HashFunction<VectorExtension>::apply(
                        load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(p_KeyPtr),
                        hashState
                     ),
                     p_ResizerState
                  ),
                  alignerState
               )
            );
            for(size_t pos = 0; pos < vector_element_count::value; ++pos) {
               base_t * posPtr = p_Data + (hashArray[pos]);
               base_t key = *(p_KeyPtr++);
               if(!
                  lookup_processing_unit(
                     key,
                     posPtr,
                     searcherState
                  )
               )
                  insert_processing_unit(key, posPtr);
            }
         }
      }
   };

}
#endif //MORPHSTORE_SET_STRATEGIES_H
