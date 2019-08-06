/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/

/**
 * @file join.h
 * @brief The template-based interfaces of join-operators. So far, there is
 * only a nested-loop-join.
 * @todo Harmonize interface names for scalar reference and extraordinary vectorized variants.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_JOIN_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_JOIN_H

#include <core/storage/column.h>
#include <core/utils/basic_types.h>

#include <vector/complex/hash.h>
#include <vector/datastructures/hash_based/strategies/linear_probing.h>
#include <vector/datastructures/hash_based/hash_utils.h>
#include <vector/datastructures/hash_based/hash_map.h>
#include <vector/datastructures/hash_based/hash_set.h>

#include <tuple>

namespace morphstore {
    
/**
 * Nested-loop-join-operator. Joins the two given columns and returns the
 * matching pairs of positions in the form of two position columns of the same
 * length.
 * 
 * Example:
 * - inDataLCol: [22, 44, 11, 22, 55, 77]
 * - inDataRCol: [33, 22, 22, 11]
 * - outPosLCol: [0, 0, 2, 3, 3]
 * - outPosRCol: [1, 2, 3, 1, 2]
 * 
 * @param inDataLCol The left column to join on.
 * @param inDataRCol The right column to join on.
 * @param outCountEstimate An optional estimate of the number of data
 * elements in each of the output columns. If specified, both output columns
 * will allocate enough memory for exactly this number of data elements.
 * Otherwise, a pessimistic estimation will be done.
 * @return A tuple of two columns containing equally many data elements. The
 * first (second) output column contains positions referring to the left
 * (right) input column. The i-th positions in the two output columns denote
 * one matching pair.
 * @todo Maybe the nested-loop-join should have a template for the join
 * predicate/operation just like the selection.
 */
template<
        class t_vector_extension,
        class t_out_pos_l_f,
        class t_out_pos_r_f,
        class t_in_data_l_f,
        class t_in_data_r_f
>
const std::tuple<
        const column<t_out_pos_l_f> *,
        const column<t_out_pos_r_f> *
>
nested_loop_join(
        const column<t_in_data_l_f> * const inDataLCol,
        const column<t_in_data_r_f> * const inDataRCol,
        const size_t outCountEstimate = 0
);

/**
 * Left-semi-N:1-nested-loop-join-operator. Joins the two given columns and
 * returns the positions of the left input's data elements having a join
 * partner in the rigth input. Every data element in the left input is assumed
 * to have at most one match in the rigth input (N:1), i.e., the right input is
 * assumed to be unique.
 * 
 * Example:
 * - inDataLCol: [11, 22, 33, 11, 44, 55]
 * - inDataRCol: [22, 22, 33, 44, 33]
 * - outPosLCol: [     1,  2,      4]
 * 
 * @param inDataLCol The left column to join on.
 * @param inDataRCol The right column to join on, containing only unique data
 * elements.
 * @param outCountEstimate An optional estimate of the number of data
 * elements in the output column. If specified, the output column will allocate
 * enough memory for exactly this number of data elements. Otherwise, a
 * pessimistic estimation will be done.
 * @return A column containing the positions (referring to the left input) of
 * all data elements in the left input which have a match in the right input.
 */
template<
        class t_vector_extension,
        class t_out_pos_l_f,
        class t_in_data_l_f,
        class t_in_data_r_f
>
const column<t_out_pos_l_f> *
left_semi_nto1_nested_loop_join(
        const column<t_in_data_l_f> * const inDataLCol,
        const column<t_in_data_r_f> * const inDataRCol,
        const size_t outCountEstimate = 0
);


//vectorized part
template<
   class VectorExtension,
   class DataStructure,
   class OutFormatCol,
   class InFormatLCol,
   class InFormatRCol
>
struct semi_equi_join_t /* {
   static
   column< OutFormatCol > const *
   apply(
      column< InFormatLCol > const * const p_InDataLCol,
      column< InFormatRCol > const * const p_InDataRCol,
      size_t const outCountEstimate = 0
   ) = delete;
}*/;


template<
   class VectorExtension,
   class DataStructure,
   class OutFormatLCol,
   class OutFormatRCol,
   class InFormatLCol,
   class InFormatRCol
>
struct natural_equi_join_t/* {
   static
   std::tuple<
      column< OutFormatLCol > const *,
      column< OutFormatRCol > const *
   > const
   apply(
      column< InFormatLCol > const * const p_InDataLCol,
      column< InFormatRCol > const * const p_InDataRCol,
      size_t const outCountEstimate = 0
   ) = delete;
}*/;



template<
   class VectorExtension,
   class OutFormatLCol,
   class OutFormatRCol,
   class InFormatLCol,
   class InFormatRCol
>
std::tuple<
   column< OutFormatLCol > const *,
   column< OutFormatRCol > const *
> const
join(
   column< InFormatLCol > const * const p_InDataLCol,
   column< InFormatRCol > const * const p_InDataRCol,
   size_t const outCountEstimate = 0
) {
   return natural_equi_join_t<
      VectorExtension,
      vectorlib::hash_map<
         VectorExtension,
         vectorlib::multiply_mod_hash,
         vectorlib::size_policy_hash::EXPONENTIAL,
         vectorlib::scalar_key_vectorized_linear_search,
         60
      >,
      OutFormatLCol,
      OutFormatRCol,
      InFormatLCol,
      InFormatRCol
   >::apply(p_InDataLCol,p_InDataRCol,outCountEstimate);
}


template<
   class VectorExtension,
   class OutFormatCol,
   class InFormatLCol,
   class InFormatRCol
>
column<OutFormatCol> const *
semi_join(
   column< InFormatLCol > const * const p_InDataLCol,
   column< InFormatRCol > const * const p_InDataRCol,
   size_t const outCountEstimate = 0
) {
   return semi_equi_join_t<
      VectorExtension,
      vectorlib::hash_set<
         VectorExtension,
         vectorlib::multiply_mod_hash,
         vectorlib::size_policy_hash::EXPONENTIAL,
         vectorlib::scalar_key_vectorized_linear_search,
         60
      >,
      OutFormatCol,
      InFormatLCol,
      InFormatRCol
   >::apply(p_InDataLCol,p_InDataRCol, outCountEstimate);
}


}
#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_JOIN_H
