//
// Created by jpietrzyk on 10.05.19.
//

#ifndef MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_SET_H
#define MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_SET_H

#include <core/memory/mm_glob.h>
#include <core/utils/preprocessor.h>

#include "../../vector_primitives.h"

#include "hash_utils.h"

#ifdef MSV_NO_SELFMANAGED_MEMORY
#include <core/memory/management/utils/alignment_helper.h>
#endif

#include <utility> //pair
#include <cstdint> //uint8_t
#include <cstddef> //size_t
#include <algorithm>

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
   class hash_set{
      public:
         template< class VectorExtension >
         MSV_CXX_ATTRIBUTE_FORCE_INLINE
         void
         insert(
            typename VectorExtension::vector_t const & p_KeysToLookup,
            typename
            LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_single_key_t
            & p_LookupInsertStrategyState
         ) {
            LookupInsertStrategy<VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH>::insert(
               p_KeysToLookup,
               p_LookupInsertStrategyState
            );
         }

         template< class VectorExtension >
         MSV_CXX_ATTRIBUTE_FORCE_INLINE
         std::pair< typename VectorExtension::mask_t, uint8_t >
         lookup(
            typename VectorExtension::vector_t const & p_KeysToLookup,
            typename
            LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_single_key_t
            & p_LookupInsertStrategyState
         ) {
            return LookupInsertStrategy<VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH>::lookup(
               p_KeysToLookup,
               p_LookupInsertStrategyState
            );
         }


      private:
         size_helper<BiggestSupportedVectorExtension, MaxLoadfactor, SPH> const m_SizeHelper;
#ifdef MSV_NO_SELFMANAGED_MEMORY
         void * const m_DataUnaligned;
#endif
         typename BiggestSupportedVectorExtension::base_t * const m_Data;
      public:
         hash_set(
            size_t const p_DistinctElementCountEstimate
         ) :
            m_SizeHelper{
               p_DistinctElementCountEstimate
            },
#ifdef MSV_NO_SELFMANAGED_MEMORY
            m_DataUnaligned{
               malloc( get_size_with_alignment_padding( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) ) )
            },
            m_Data{ ( typename BiggestSupportedVectorExtension::base_t * ) create_aligned_ptr( m_DataUnaligned ) }
#else
            m_Data{ ( typename BiggestSupportedVectorExtension::base_t * )
                       malloc( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) ) }
#endif
         {
            std::fill(m_Data, m_Data+m_SizeHelper.m_Count, 0);
         }


         typename BiggestSupportedVectorExtension::base_t * get_data( void ) {
            return m_Data;
         }

         size_t get_bucket_count( void ) {
            return m_SizeHelper.m_Count;
         }

         template< class VectorExtension >
         using strategy_state =
         typename
         LookupInsertStrategy<
            VectorExtension,
            BiggestSupportedVectorExtension,
            HashFunction,
            SPH
         >::state_single_key_t;


         template< class VectorExtension >
         strategy_state<VectorExtension>
         get_lookup_insert_strategy_state( void ) {
            return
               typename
               LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_single_key_t(
                  m_Data,
                  m_SizeHelper.m_Count
               );
         }



         ~hash_set() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
            free( m_DataUnaligned );
#else
            free( m_Data );
#endif
         }

         void print( void ) const {
            uint64_t mulres, resizeres, alignres;
            fprintf( stdout, "HashSet idx;Key;Key*Prime;Resized;Aligned (StartPos)\n");
            for( size_t i = 0; i < m_SizeHelper.m_Count; ++i ) {
               __builtin_umull_overflow( m_Data[i], 65537, &mulres);
               resizeres = mulres & 1023;
               alignres = resizeres & (typename BiggestSupportedVectorExtension::base_t)~(BiggestSupportedVectorExtension::vector_helper_t::element_count::value - 1);
               fprintf( stdout, "%lu;%lu;%lu;%lu,%lu\n", i, m_Data[i],mulres,resizeres,alignres );
            }
         }
   };



}

#endif //MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_SET_H
