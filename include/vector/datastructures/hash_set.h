//
// Created by jpietrzyk on 10.05.19.
//

#ifndef MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_SET_H
#define MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_SET_H

#include <core/memory/mm_glob.h>
#include <core/storage/column.h>
#include <core/morphing/format.h>
#include <vector/general_vector.h>
#include <vector/complex/hash.h>
#include <vector/primitives/io.h>
#include <vector/primitives/create.h>
#include <vector/primitives/calc.h>
#include <vector/primitives/compare.h>


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
      size_t MaxLoadfactor //60 if 0.6...
   >
   class hash_set_lpcs {

      private:
         /**
          * 0   [ 0 ]   ==>
          * 1   [ 0 ]
          * 2   [ 0 ]
          * 3   [ 0 ]
          * 4   [ 0 ]
          * 5   [ 0 ]
          * 6   [ 0 ]
          *
          * @tparam VectorExtension
          * @param p_Key
          * @param p_StartPosition
          * @param p_LastPossiblePosition
          */
         template< class VectorExtension >
         MSV_CXX_ATTRIBUTE_FORCE_INLINE
         void build_processing_unit(
            typename VectorExtension::base_t const p_Key,
            typename VectorExtension::base_t * p_StartPosition,
            typename VectorExtension::base_t const * const p_LastPossiblePosition,
            typename const & p_State
         ) {
            IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
            vector_t const zeroVec = set1<VectorExtension>(0);
            vector_t const & keyVec = set1<VectorExtension>(p_Key);
            mask_t searchOffset = 0;
            base_t * current;
            base_t * next = ( p_StartPosition > p_LastPossiblePosition ) ? p_LastPossiblePosition : p_StartPosition;
            do {
               current = next;
               vector_t searchVec = load<VectorExtension, iov::UNALIGNED, vector_size_bit::value>( current );
               if(
                  equal<VectorExtension>::apply(
                     keyVec,
                     searchVec
                  ) != 0
               ) {
                  return;
               }
               searchOffset =
                  equal<VectorExtension>::apply(
                     zeroVec,
                     searchVec
                  );
               next = (next<p_LastPossiblePosition) ? next + vector_element_count : m_Data;
            }while( searchOffset == 0 );
            *(current + __builtin_ctz(searchOffset)) = p_Key;
         }

         template<class Format, class VectorExtension>
         void build_batch(
            typename VectorExtension::base_t const *& p_DataPtr,
            size_t const p_Count
         ) {
            IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
            MSV_CXX_ATTRIBUTE_HOT typename HashFunction<VectorExtension>::state_t hashState(m_Size);
            MSV_CXX_ATTRIBUTE_HOT alignas(vector_size_byte::value) base_t hashArray[vector_element_count::value];

            base_t * lastPossiblePointer = m_End;
            // we need to normalize the endptr. if vector registers are used the end pointer has to be
            // reduced by the size of the vector register
            if( vector_element_count != 1 ) {
               lastPossiblePointer -= vector_element_count;
            }

            for(size_t i = 0; i < p_Count; ++i) {
               store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
                  hashArray,
                  //1. the data which should be inserted is loaded into a vector register.
                  //2. The keys were hashed using the specified Hashfunction
                  //3. The hashes are NOT normalized (they can be greater then the maximum size)
                  HashFunction<VectorExtension>::apply(
                     load<VectorExtension, iov::ALIGNED, vector_size_bit::value>(p_DataPtr),
                     hashState
                  )
               );
               for(size_t pos = 0; pos < vector_element_count::value; ++pos) {
                  build_processing_unit<VectorExtension>(
                     *(p_DataPtr++),
                     m_Data + (hashArray[pos] % m_Size),
                     lastPossiblePointer,
                     hashState
                  );
               }
            }
         }


      public:
         template<class Format, class VectorExtension>
         void build(
            morphstore::column<Format> const * const p_Column
         ) {
            const size_t inDataCount = p_Column->get_count_values();
            base_t const * inDataPtr = p_Column->get_data( );


            size_t const vectorCount = inDataCount / vector_element_count::value;
            size_t const remainderCount = inDataCount % vector_element_count::value;



         }

      private:
         size_t const m_Size;
         base_t * const m_Data;
         base_T * const m_End;
      public:
/*
         template< typename ... HashFunctionCtorArguments >
         hash_set_lpcs(size_t const p_Size, HashFunctionCtorArguments&&... p_HashArgs) :
            m_size{ p_Size + p_Size * MaxLoadfactor },
            m_Data{ ( base_t * ) malloc( m_Size * sizeof( T ) ) },
            hash_fn{ std::forward<HashFunctionCtorArguments>(p_HashArgs)... }{ }
*/
         template<class Format, class VectorExtension>
         hash_set_lpcs(
            morphstore::column<Format> const * const p_Column,
            size_t const p_DistinctElementCountEstimate = 0
         ) :
            m_Size{
               bool(p_DistinctElementCountEstimate) ?
                  (p_DistinctElementCountEstimate * sizeof(base_t)) +
                  (p_DistinctElementCountEstimate * sizeof(base_t)) * 100 / MaxLoadfactor
               : p_Column->get_size_used_byte() + p_Column->get_size_used_byte() * MaxLoadfactor;
            },
            m_Data{ ( base_t * ) malloc( m_Size * sizeof( base_t ) ) },
            m_End{ m_Data + m_Size } {

            build<Format, VectorExtension>( p_Column );

         }

         ~hash_set_lpcs() {
            free( m_Data );
         }
   };



}

#endif //MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_SET_H
