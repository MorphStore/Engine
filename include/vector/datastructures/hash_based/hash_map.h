//
// Created by jpietrzyk on 10.05.19.
//

#ifndef MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_MAP_H
#define MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_MAP_H

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
   class hash_map{
   public:
      using strategy MSV_CXX_ATTRIBUTE_PPUNUSED = LookupInsertStrategy;
   public:

      template< class VectorExtension >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      void
      insert(
         typename VectorExtension::vector_t const & p_KeysToLookup,
         typename VectorExtension::vector_t const & p_Values,
         typename
         LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_single_key_single_value_t
         & p_LookupInsertStrategyState
      ) {
         LookupInsertStrategy<VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH>::insert(
            p_KeysToLookup,
            p_Values,
            p_LookupInsertStrategyState
         );
      }

      template< class VectorExtension >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      std::tuple<
         typename VectorExtension::vector_t,      // groupID vector register
         typename VectorExtension::vector_t,      // groupExt vector register
         typename VectorExtension::mask_t, // active groupExt elements
         uint8_t        // Number of active groupExt elements
      >
      insert_and_lookup(
         typename VectorExtension::vector_t const & p_InKeyVector,
         typename VectorExtension::base_t & p_InStartPosFromKey,
         typename VectorExtension::base_t & p_InStartValue,
         typename LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_single_key_single_value_t
         & p_LookupInsertStrategyState
      ) {
         return LookupInsertStrategy<VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH>::insert_and_lookup(
            p_InKeyVector,
            p_InStartPosFromKey,
            p_InStartValue,
            p_LookupInsertStrategyState
         );
      }


      template< class VectorExtension >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      std::tuple< typename VectorExtension::vector_t, typename VectorExtension::mask_t, uint8_t >
      lookup(
         typename VectorExtension::vector_t const & p_KeysToLookup,
         typename
         LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_single_key_single_value_t
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
      void * const m_KeysUnaligned;
      void * const m_ValuesUnaligned;
#endif
      typename BiggestSupportedVectorExtension::base_t * const m_Keys;
      typename BiggestSupportedVectorExtension::base_t * const m_Values;
   public:
      hash_map(
         size_t const p_DistinctElementCountEstimate
      ) :
         m_SizeHelper{
            p_DistinctElementCountEstimate
         },
#ifdef MSV_NO_SELFMANAGED_MEMORY
         m_KeysUnaligned{
            malloc( get_size_with_alignment_padding( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) ) )
         },
         m_ValuesUnaligned{
            malloc( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) )
         },
         m_Keys{ ( typename BiggestSupportedVectorExtension::base_t * ) create_aligned_ptr( m_KeysUnaligned ) },
         m_Values{ ( typename BiggestSupportedVectorExtension::base_t * ) create_aligned_ptr( m_ValuesUnaligned ) }
#else
         m_Keys{
            ( typename BiggestSupportedVectorExtension::base_t * )
            malloc( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) )
         },
         m_Values{
            ( typename BiggestSupportedVectorExtension::base_t * )
            malloc( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) )
         }
#endif
      {
         std::fill(m_Keys, m_Keys+m_SizeHelper.m_Count, 0);
         std::fill(m_Values, m_Values+m_SizeHelper.m_Count, 0);
      }


      typename BiggestSupportedVectorExtension::base_t * get_data_keys( void ) {
         return m_Keys;
      }

      typename BiggestSupportedVectorExtension::base_t * get_data_values( void ) {
         return m_Values;
      }

      size_t get_bucket_count( void ) {
         return m_SizeHelper.m_Count;
      }

      template< class VectorExtension >
      typename
      LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_single_key_single_value_t
      get_lookup_insert_strategy_state( void ) {
         return
            typename
            LookupInsertStrategy< VectorExtension, BiggestSupportedVectorExtension, HashFunction, SPH >::state_single_key_single_value_t(
               m_Keys,
               m_Values,
               m_SizeHelper.m_Count
            );
      }



      ~hash_map() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
         free( m_ValuesUnaligned );
         free( m_KeysUnaligned );
#else
         free( m_Values );
         free( m_Keys );

#endif
      }

      void print( void ) const {
         uint64_t mulres, resizeres, alignres;
         fprintf( stdout, "HashSet idx;Key;Key*Prime;Resized;Aligned (StartPos)\n");
         for( size_t i = 0; i < m_SizeHelper.m_Count; ++i ) {
            __builtin_umull_overflow( m_Keys[i], 65537, &mulres);
            resizeres = mulres & 1023;
            alignres = resizeres & (typename BiggestSupportedVectorExtension::base_t)~(BiggestSupportedVectorExtension::vector_helper_t::element_count::value - 1);
            fprintf( stdout, "%lu;%lu;%lu;%lu,%lu\n", i, m_Keys[i],mulres,resizeres,alignres );
         }
      }
   };



}

#endif //MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_MAP_H
