//
// Created by jpietrzyk on 10.05.19.
//

#ifndef MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_BINARY_KEY_MAP_H
#define MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_BINARY_KEY_MAP_H

#include <core/memory/mm_glob.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_primitives.h>
#include <vector/datastructures/hash_based/hash_utils.h>

#ifdef MSV_NO_SELFMANAGED_MEMORY
#include <core/memory/management/utils/alignment_helper.h>
#endif

#include <tuple> //tuple
#include <cstdint> //uint8_t
#include <cstddef> //size_t
#include <algorithm> //std::fill

namespace vectorlib {
   /**
    * Hash set constant size (NO RESIZING), linear probing
    * @tparam VectorExtension
    * @tparam HashFunction
    * @tparam MaxLoadfactor
    */
   template<
      class BiggestSupportedVectorExtension,
      template<class> class HashFunction,
      size_policy_hash SPH,
      template<class, class, template<class>class, size_policy_hash> class LookupInsertStrategy,
      size_t MaxLoadfactor //60 if 0.6...
   >
   class hash_binary_key_map{
      public:
         template< class VectorExtension >
         MSV_CXX_ATTRIBUTE_FORCE_INLINE
         std::tuple<
            typename VectorExtension::vector_t,      // groupID vector register
            typename VectorExtension::vector_t,      // groupExt vector register
            typename VectorExtension::mask_t, // active groupExt elements
            typename VectorExtension::mask_size_t        // Number of active groupExt elements
         >
         insert_and_lookup(
            typename VectorExtension::vector_t const & p_KeysFirstToLookup,
            typename VectorExtension::vector_t const & p_KeysSecondToLookup,
            typename VectorExtension::base_t & p_InStartPosFromKey,
            typename VectorExtension::base_t & p_InStartValue,
            typename LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_double_key_single_value_t &
            p_LookupInsertStrategyState
         ) {
            return LookupInsertStrategy<VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH>::insert_and_lookup(
               p_KeysFirstToLookup,
               p_KeysSecondToLookup,
               p_InStartPosFromKey,
               p_InStartValue,
               p_LookupInsertStrategyState
            );
         }

      private:
         size_helper<BiggestSupportedVectorExtension, MaxLoadfactor, SPH> const m_SizeHelper;
#ifdef MSV_NO_SELFMANAGED_MEMORY
         void * const m_KeysFirstUnaligned;
         void * const m_KeysSecondUnaligned;
         void * const m_ValuesUnaligned;
#endif
         typename BiggestSupportedVectorExtension::base_t * const m_KeysFirst;
         typename BiggestSupportedVectorExtension::base_t * const m_KeysSecond;
         typename BiggestSupportedVectorExtension::base_t * const m_Values;
      public:
         hash_binary_key_map(
            size_t const p_DistinctElementCountEstimate
         ) :
            m_SizeHelper{
               p_DistinctElementCountEstimate
            },

#ifdef MSV_NO_SELFMANAGED_MEMORY
            m_KeysFirstUnaligned{
               malloc( get_size_with_alignment_padding( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) ) )
            },
            m_KeysSecondUnaligned{
               malloc( get_size_with_alignment_padding( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) ) )
            },
            m_ValuesUnaligned{
               malloc( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) )
            },
            m_KeysFirst{ ( typename BiggestSupportedVectorExtension::base_t * ) create_aligned_ptr( m_KeysFirstUnaligned ) },
            m_KeysSecond{ ( typename BiggestSupportedVectorExtension::base_t * ) create_aligned_ptr( m_KeysSecondUnaligned ) },
            m_Values{ ( typename BiggestSupportedVectorExtension::base_t * ) create_aligned_ptr( m_ValuesUnaligned ) }
#else
            m_KeysFirst{
               ( typename BiggestSupportedVectorExtension::base_t * )
                  malloc( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) ) },
            m_KeysSecond{
               ( typename BiggestSupportedVectorExtension::base_t * )
                  malloc( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) ) },
            m_Values{
               ( typename BiggestSupportedVectorExtension::base_t * )
                  malloc( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) ) }
#endif
         {
            std::fill(m_KeysFirst, m_KeysFirst+m_SizeHelper.m_Count, 0);
            std::fill(m_KeysSecond, m_KeysSecond+m_SizeHelper.m_Count, 0);
            std::fill(m_Values, m_Values+m_SizeHelper.m_Count, 0);
         }


         typename BiggestSupportedVectorExtension::base_t * get_data_keys_first( void ) {
            return m_KeysFirst;
         }

         typename BiggestSupportedVectorExtension::base_t * get_data_keys_second( void ) {
            return m_KeysSecond;
         }

         typename BiggestSupportedVectorExtension::base_t * get_data_values( void ) {
            return m_Values;
         }

         size_t get_bucket_count( void ) {
            return m_SizeHelper.m_Count;
         }

         template< class VectorExtension >
         typename
         LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_double_key_single_value_t
         get_lookup_insert_strategy_state( void ) {
            return
               typename
               LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_double_key_single_value_t(
                  m_KeysFirst,
                  m_KeysSecond,
                  m_Values,
                  m_SizeHelper.m_Count
               );
         }



         ~hash_binary_key_map() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
            free( m_ValuesUnaligned );
            free( m_KeysSecondUnaligned );
            free( m_KeysFirstUnaligned );
#else
            free( m_Values );
            free( m_KeysSecond );
            free( m_KeysFirst );
#endif
         }
   };

}

#endif //MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_BINARY_KEY_MAP_H