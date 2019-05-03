#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_CALC_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_CALC_UNCOMPR_H

#include <vector/general_vector.h>
namespace morphstore {

   template<template<typename> class t_unary_op>
   struct calc_unary<
      t_unary_op,
      processing_style_t::vec256,
      uncompr_f,
      uncompr_f
   > {
      static
      const column <uncompr_f> *apply(
         const column <uncompr_f> *const inDataCol
      ) {


      }
   };


   template<class VectorExtension>
   const column <uncompr_f> *
      agg_sum(
      column < uncompr_f >
   const * const p_DataColumn
   ) {

   }


}
#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_CALC_UNCOMPR_H
