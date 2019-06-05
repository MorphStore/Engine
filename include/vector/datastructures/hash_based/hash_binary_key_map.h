//
// Created by jpietrzyk on 10.05.19.
//

#ifndef MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_BINARY_KEY_MAP_H
#define MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_MAP_H

#include <core/memory/mm_glob.h>
#include <core/utils/preprocessor.h>
#include <vector/general_vector.h>
#include <vector/datastructures/hash_based/hash_utils.h>

#include <tuple> //tuple
#include <cstdint> //uint8_t
#include <cstddef> //size_t
#include <algorithm> //std::fill

namespace vector {
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
            uint8_t        // Number of active groupExt elements
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
            m_KeysFirst{
               ( typename BiggestSupportedVectorExtension::base_t * )
                  malloc( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) ) },
            m_KeysSecond{
               ( typename BiggestSupportedVectorExtension::base_t * )
                  malloc( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) ) },
            m_Values{
               ( typename BiggestSupportedVectorExtension::base_t * )
                  malloc( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) ) } {
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
            free( m_Values );
            free( m_KeysSecond );
            free( m_KeysFirst );
         }
   };

}

#endif //MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_BINARY_KEY_MAP_H