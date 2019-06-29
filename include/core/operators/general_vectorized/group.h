//
// Created by jpietrzyk on 29.06.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_GROUP_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_GROUP_H

#include <core/storage/column.h>
#include <core/utils/basic_types.h>
//@todo: Fix group interface!!!
//#include <core/operators/interfaces/group.h>

#include <vector/complex/hash.h>
#include <vector/datastructures/hash_based/hash_utils.h>
#include <vector/datastructures/hash_based/hash_map.h>
#include <vector/datastructures/hash_based/hash_binary_key_map.h>
#include <vector/datastructures/hash_based/strategies/linear_probing.h>


#include <tuple>

namespace morphstore {

   template<
      class VectorExtension,
      class DataStructure,
      class OutFormatGroupIds,
      class OutFormatGroupExtents,
      class InFormatData
   >
   struct group_unary_wit_t;

   template<
      class VectorExtension,
      class DataStructure,
      class OutFormatGroupIds,
      class OutFormatGroupExtents,
      class InFormatGroupIdCol,
      class InFormatDataCol
   >
   struct group_binary_wit_t;


   template<
      class VectorExtension,
      class DataStructure,
      class OutFormatGroupIds,
      class OutFormatGroupExtents,
      class InFormatData
   >
   const std::tuple<
      const column<OutFormatGroupIds> *,
      const column<OutFormatGroupExtents> *
   > group_vec(
      column<InFormatData> const * const  p_InDataCol,
      size_t const outCountEstimate = 0
   ) {
      return group_unary_wit_t<
         VectorExtension,
         vector::hash_map<
            VectorExtension,
            vector::multiply_mod_hash,
            vector::size_policy_hash::EXPONENTIAL,
            vector::scalar_key_vectorized_linear_search,
            60
         >,
         OutFormatGroupIds,
         OutFormatGroupExtents,
         InFormatData
         >::apply(p_InDataCol,outCountEstimate);
   }

   template<
      class VectorExtension,
      class DataStructure,
      class OutFormatGroupIds,
      class OutFormatGroupExtents,
      class InFormatGroupIdCol,
      class InFormatDataCol
   >
   const std::tuple<
      const column<OutFormatGroupIds> *,
      const column<OutFormatGroupExtents> *
   > group_vec(
      column<InFormatGroupIdCol> const * const p_InGrCol,
      column<InFormatDataCol> const * const p_InDataCol,
      size_t const outCountEstimate = 0
   ) {
      return group_binary_wit_t<
         VectorExtension,
         vector::hash_binary_key_map<
            VectorExtension,
            vector::multiply_mod_hash,
            vector::size_policy_hash::EXPONENTIAL,
            vector::scalar_key_vectorized_linear_search,
            60
         >,
         OutFormatGroupIds,
         OutFormatGroupExtents,
         InFormatGroupIdCol,
         InFormatDataCol
         >::apply(p_InGrCol,p_InDataCol,outCountEstimate);
   }
}

#include <core/operators/general_vectorized/group_uncompr.h>
#include <core/operators/general_vectorized/group_compr.h>

#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_GROUP_H
