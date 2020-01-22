//
// Created by jpietrzyk on 26.04.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_UNCOMPR_H

#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

namespace morphstore {

    using namespace vectorlib;
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
         vector_t const predicateVector = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(p_Predicate);
         vector_t positionVector = vectorlib::set_sequence<VectorExtension, vector_base_t_granularity::value>(startid,1);
         vector_t const addVector = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(vector_element_count::value);
         for(size_t i = 0; i < p_Count; ++i) {
            vector_t dataVector = vectorlib::load<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(p_DataPtr);
            vector_mask_t resultMask =
               select_processing_unit<VectorExtension,Operator>::apply(
                  dataVector,
                  predicateVector
               );
            vectorlib::compressstore<VectorExtension, vectorlib::iov::UNALIGNED, vector_base_t_granularity::value>(p_OutPtr, positionVector, resultMask);
            positionVector = vectorlib::add<VectorExtension, vector_base_t_granularity::value>::apply(positionVector,addVector);

            //p_OutPtr += __builtin_popcount( resultMask );
            p_OutPtr += vectorlib::count_matches<VectorExtension>::apply( resultMask );
            p_DataPtr += vector_element_count::value;
         }
      }
   };


  template<int granularity, typename T, template< class, int > class Operator>
  struct call_scalar_batch_select;

  template<typename T, template< class, int > class Operator>
  struct call_scalar_batch_select<64, T, Operator>{
    IMPORT_VECTOR_BOILER_PLATE(scalar<v64<uint64_t>>)
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    static void call(     base_t const *& p_DataPtr,
         base_t const p_Predicate,
         base_t *& p_OutPtr,
         size_t const p_Count,
         int startid = 0){
         select_batch<scalar<v64<T>>, Operator>::apply(p_DataPtr, p_Predicate, p_OutPtr, p_Count,startid);
    }
  };


  template<typename T, template< class, int > class Operator>
  struct call_scalar_batch_select<32, T, Operator>{
    IMPORT_VECTOR_BOILER_PLATE(scalar<v32<uint32_t>>)

    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    static void call(    base_t const *& p_DataPtr,
         base_t const p_Predicate,
         base_t *& p_OutPtr,
         size_t const p_Count,
         int startid = 0){
        select_batch<scalar<v32<T>>,Operator>::apply(p_DataPtr, p_Predicate, p_OutPtr, p_Count,startid);
    }
  };

  template<typename T, template< class, int > class Operator>
  struct call_scalar_batch_select<16, T, Operator>{
    IMPORT_VECTOR_BOILER_PLATE(scalar<v16<uint16_t>>)

    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    static void call(    base_t const *& p_DataPtr,
         base_t const p_Predicate,
         base_t *& p_OutPtr,
         size_t const p_Count,
         int startid = 0){
        select_batch<scalar<v16<T>>,Operator>::apply(p_DataPtr, p_Predicate, p_OutPtr, p_Count,startid);
    }
  };

  template<typename T, template< class, int > class Operator>
  struct call_scalar_batch_select<8, T, Operator>{
    IMPORT_VECTOR_BOILER_PLATE(scalar<v8<uint8_t>>)

    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    static void call(    base_t const *& p_DataPtr,
         base_t const p_Predicate,
         base_t *& p_OutPtr,
         size_t const p_Count,
         int startid = 0){
        select_batch<scalar<v8<T>>,Operator>::apply(p_DataPtr, p_Predicate, p_OutPtr, p_Count,startid);
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


        call_scalar_batch_select<vector_base_t_granularity::value,typename VectorExtension::base_t, Operator>::call(inDataPtr, p_Predicate, outDataPtr, remainderCount,vectorCount*vector_element_count::value);

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
