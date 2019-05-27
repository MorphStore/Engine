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
#include <vector/datastructures/set_utils.h>
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
      size_policy_set SPS,
      template<class, class, template<class>class, size_policy_set> class LookupInsertStrategy,
      size_t MaxLoadfactor //60 if 0.6...
   >
   class hash_set{
      public:
         template<class Format, class VectorExtension>
         void build(
            morphstore::column<Format> const * const p_Column
         ) {
            IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
            typename key_resizer< VectorExtension, SPS >::state_t resizerState( m_SizeHelper.m_Count );
            const size_t inDataCount = p_Column->get_count_values();
            base_t * inDataPtr = p_Column->get_data( );

            size_t const vectorCount = inDataCount / vector_element_count::value;
            size_t const remainderCount = inDataCount % vector_element_count::value;

            LookupInsertStrategy<
               VectorExtension,
               BiggestSupportedVectorExtension,
               HashFunction,
               SPS
            >::build_batch( inDataPtr, m_Data, vectorCount, resizerState);

            /*LookupInsertStrategy<
               scalar<base_t> ,
               BiggestSupportedVectorExtension,
               HashFunction,
               SPS
            >::build_batch( inDataPtr, m_Data, vectorCount, m_Size);*/
         }

      private:
         key_size_helper<MaxLoadfactor, SPS> const m_SizeHelper;
         typename BiggestSupportedVectorExtension::base_t * const m_Data;
      public:
         hash_set(
            //this is the estimation of distinct elements within the column
            size_t const p_DistinctElementCountEstimate
         ) :
            m_SizeHelper{
//               bool(p_DistinctElementCountEstimate) ?
               p_DistinctElementCountEstimate
//               : p_Column->get_count_values()
            },
            m_Data{
               ( typename BiggestSupportedVectorExtension::base_t * )
               malloc( m_SizeHelper.m_Count * sizeof( typename BiggestSupportedVectorExtension::base_t ) ) } {

         }


         typename BiggestSupportedVectorExtension::base_t * get_data( void ) {
            return m_Data;
         }

         size_t get_bucket_count( void ) {
            return m_SizeHelper.m_Count;
         }

         ~hash_set() {
            free( m_Data );
         }
   };



}

#endif //MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_SET_H
