//
// Created by jpietrzyk on 26.04.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_UNCOMPR_H

#include <vector/general_vector.h>
#include <vector/primitives/io.h>
#include <vector/primitives/create.h>
#include <vector/primitives/compare.h>
#include <vector/primitives/calc.h>
#include <core/utils/preprocessor.h>

namespace morphstore {

   template<class VectorExtension, template<class> class Comparator>
   struct select_processing_unit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_mask_t const &
      apply(
         vector_t const & p_DataVector
         vector_t const & p_PredicateVector
      ) {
         return Comparator<VectorExtension>::apply(
            p_DataVector,
            p_PredicateVector
         );
      }
   };

   //@todo: SCALAR SEEMS TO BE SUPER INEFFICIENT, because of __builtin_popcount
   template<class VectorExtension, template<class> class Comparator>
   struct select_batch {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
         base_t const * p_DataPtr,
         base_t const p_Predicate,
         base_t *& p_OutPtr,
         size_t const p_Count
      ) {
         vector_t const predicateVector = vector::set1<VectorExtension, vector_base_t_granularity::value>(p_Predicate);
         vector_t positionVector = vector::set_sequence<VectorExtension, vector_base_t_granularity::value>(0,1);
         vector_t const addVector = vector::set1<VectorExtension, vector_base_t_granularity::value>(vector_element_count::value);
         for(size_t i = 0; i < p_Count; ++i) {
            vector_t dataVector = vector::load<VectorExtension, vector::iov::ALIGNED, vector_size_bit::value>(p_DataPtr);
            vector_mask_t resultMask =
               select_processing_unit<VectorExtension,Comparator>::apply(
                  dataVector,
                  predicateVector
               );
            vector::compressstore<VectorExtension, vector::iov::UNALIGNED, vector_size_bit::value>(p_OutPtr, positionVector, resultMask);
            positionVector = vector::add<VectorExtension, vector_base_t_granularity::value>::apply(positionVector,addVector);

            p_OutPtr += __builtin_popcount( resultMask );
            p_DataPtr += vector_element_count::value;
         }
      }
   };

   template<class VectorExtension, template<class> class Comparator>
   struct select {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static
      column<uncompr_f> const *
      apply(
         column< uncompr_f > const * const p_DataColumn,
         base_t const p_Predicate,
         const size_t outPosCountEstimate = 0
      ) {


         uint64_t * outP =  outPosCol->get_data();
         //I know the following is ugly, but _mm_maskstore_epi64 requires a long long (64 bit types are only long on a 64-bit system))
         __m256i * outPos =  reinterpret_cast< __m256i * >(outP);
         const __m256i * const initOutPos = (__m256i *) (outPosCol->get_data());



         size_t const inDataCount = p_DataColumn->get_count_values();
         base_t const * const inDataPtr = p_DataColumn->get_data( );
         size_t const sizeByte =
            bool(outPosCountEstimate)
            ? (outPosCountEstimate * sizeof(base_t))
            : p_DataColumn->get_size_used_byte();

         auto outDataCol = new column<uncompr_f>(sizeByte);
         base_t * outDataPtr = outDataCol->get_data( );
         base_t * const outDataPtrOrigin = const_cast< base_t * const >(outDataPtr);

         size_t const vectorCount = inPosCount / vector_element_count::value;
         size_t const remainderCount = inPosCount % vector_element_count::value;

         base_t const * p_DataPtr,
         base_t const p_Predicate,
         base_t *& p_OutPtr,
         size_t const p_Count

         select_batch<VectorExtension, Comparator>::apply(inDataPtr, p_Predicate, outDataPtr, vectorCount);
         select_batch<vector::scalar<base_t>, Comparator>::apply(inDataPtr, p_Predicate, outDataPtr, remainderCount);

         size_t const outDataCount = outDataPtr - outDataPtrOrigin;

         outDataCol->set_meta_data(outDataCount, outDataCount*sizeof(base_t));

         return outDataCol;
      }
   };


}



#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_AGG_SUM_UNCOMPR_H


