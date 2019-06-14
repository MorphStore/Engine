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
#include <vector/scalar/extension_scalar.h>
#include <vector/scalar/primitives/calc_scalar.h>
#include <vector/scalar/primitives/compare_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/scalar/primitives/create_scalar.h>

namespace morphstore {

    using namespace vector;
   template<class VectorExtension,  template< class, int > class Operator>
   struct select_processing_unit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_mask_t 
      apply(
         vector_t const p_DataVector,
         vector_t const p_PredicateVector
      ) {
         return Operator<VectorExtension,VectorExtension::vector_helper_t::granularity::value>::apply(
            p_DataVector,
            p_PredicateVector
         );
      }
   };

   //@todo: SCALAR SEEMS TO BE SUPER INEFFICIENT, because of __builtin_popcount
   template<class VectorExtension,  template< class, int > class Operator>
   struct select_batch {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
         base_t const *& p_DataPtr,
         base_t const p_Predicate,
         base_t *& p_OutPtr,
         size_t const p_Count,
         int startid = 0
      ) {
         vector_t const predicateVector = vector::set1<VectorExtension, vector_base_t_granularity::value>(p_Predicate);
         vector_t positionVector = vector::set_sequence<VectorExtension, vector_base_t_granularity::value>(startid,1);
         vector_t const addVector = vector::set1<VectorExtension, vector_base_t_granularity::value>(vector_element_count::value);
         for(size_t i = 0; i < p_Count; ++i) {
            vector_t dataVector = vector::load<VectorExtension, vector::iov::ALIGNED, vector_size_bit::value>(p_DataPtr);
            vector_mask_t resultMask =
               select_processing_unit<VectorExtension,Operator>::apply(
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

   template<class VectorExtension, template< class, int > class Operator>
   struct select_t {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static
      column<uncompr_f> const *
      apply(
         column< uncompr_f > const * const p_DataColumn,
         base_t const p_Predicate,
         const size_t outPosCountEstimate = 0
      ) {

         size_t const inDataCount = p_DataColumn->get_count_values();
         base_t const * inDataPtr = p_DataColumn->get_data( );
         size_t const sizeByte =
            bool(outPosCountEstimate)
            ? (outPosCountEstimate * sizeof(base_t))
            : p_DataColumn->get_size_used_byte();

         auto outDataCol = new column<uncompr_f>(sizeByte);
         base_t * outDataPtr = outDataCol->get_data( );
         base_t * const outDataPtrOrigin = const_cast< base_t * const >(outDataPtr);

         size_t const vectorCount = inDataCount / vector_element_count::value;
         size_t const remainderCount = inDataCount % vector_element_count::value;

         
         select_batch<VectorExtension, Operator>::apply(inDataPtr, p_Predicate, outDataPtr, vectorCount);
         
         select_batch<scalar<v64<uint64_t>>, Operator>::apply(inDataPtr, p_Predicate, outDataPtr, remainderCount,vectorCount*vector_element_count::value);

         size_t const outDataCount = outDataPtr - outDataPtrOrigin;

         outDataCol->set_meta_data(outDataCount, outDataCount*sizeof(base_t));

         return outDataCol;
      }
   };

    template<template< class, int > class Operator, class t_vector_extension, class t_out_pos_f, class t_in_data_f>
    column<uncompr_f> const * select(
        column< uncompr_f > const * const p_DataColumn,
         typename t_vector_extension::vector_helper_t::base_t const p_Predicate,
         const size_t outPosCountEstimate = 0
      ){
        return select_t<t_vector_extension, Operator>::apply(p_DataColumn,p_Predicate,outPosCountEstimate);
    }


}



#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_AGG_SUM_UNCOMPR_H


