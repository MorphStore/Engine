//
// Created by jpietrzyk on 29.06.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_H

#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/operators/interfaces/join.h>

#include <vector/complex/hash.h>
#include <vector/datastructures/hash_based/hash_utils.h>
#include <vector/datastructures/hash_based/hash_map.h>
#include <vector/datastructures/hash_based/hash_set.h>
#include <vector/datastructures/hash_based/strategies/linear_probing.h>

#include <tuple>


namespace morphstore {

   template<
      class VectorExtension,
      class DataStructure,
      class OutFormatCol,
      class InFormatLCol,
      class InFormatRCol
   >
   struct semi_equi_join_t;


   template<
      class VectorExtension,
      class DataStructure,
      class OutFormatLCol,
      class OutFormatRCol,
      class InFormatLCol,
      class InFormatRCol
   >
   struct natural_equi_join_t;



   template<
      class VectorExtension,
      class OutFormatLCol,
      class OutFormatRCol,
      class InFormatLCol,
      class InFormatRCol
   >
   std::tuple<
      column<OutFormatLCol> const *,
      column<OutFormatRCol> const *
   > const
   join(
      column<InFormatLCol> const *const p_InDataLCol,
      column<InFormatRCol> const *const p_InDataRCol,
      size_t const outCountEstimate
   ) {
      return natural_equi_join_t<
         VectorExtension,
         vector::hash_map<
            VectorExtension,
            vector::multiply_mod_hash,
            vector::size_policy_hash::EXPONENTIAL,
            vector::scalar_key_vectorized_linear_search,
            60
         >,
         OutFormatLCol,
         OutFormatRCol,
         InFormatLCol,
         InFormatRCol
      >::apply(p_InDataLCol, p_InDataRCol, outCountEstimate);
   }


   template<
      class VectorExtension,
      class OutFormatCol,
      class InFormatLCol,
      class InFormatRCol
   >
   column<OutFormatCol> const *
   semi_join(
      column<InFormatLCol> const *const p_InDataLCol,
      column<InFormatRCol> const *const p_InDataRCol,
      size_t const outCountEstimate
   ) {
      return semi_equi_join_t<
         VectorExtension,
         vector::hash_set<
            VectorExtension,
            vector::multiply_mod_hash,
            vector::size_policy_hash::EXPONENTIAL,
            vector::scalar_key_vectorized_linear_search,
            60
         >,
         OutFormatCol,
         InFormatLCol,
         InFormatRCol
      >::apply(p_InDataLCol, p_InDataRCol, outCountEstimate);
   }

}

#include <core/operators/general_vectorized/join_uncompr.h>
#include <core/operators/general_vectorized/join_compr.h>

#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_H
