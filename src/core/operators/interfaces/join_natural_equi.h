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
 * @file join_natural_equi.h
 * @brief The template-based interface and the corresponding convenience function of natural-equi-join-operator.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_JOIN_NATURAL_EQUI_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_JOIN_NATURAL_EQUI_H

#include <core/storage/column.h>

#include <tuple>

namespace morphstore {

   /**
    * @struct natural_equi_join_t
    * @brief Helper struct for Natural-equi-join-operator.
    * @tparam t_vector_extension Used vector extension.
    * @tparam t_out_pos_l_f Compression format for the position list of the left result column.
    * @tparam t_out_pos_r_f Compression format for the position list of the right result column.
    * @tparam t_in_pos_l_f Compression format for the position list of the left input column.
    * @tparam t_in_pos_r_f Compression format for the position list of the right input column.
    */
   template<
      class t_vector_extension,
      class t_out_pos_l_f,
      class t_out_pos_r_f,
      class t_in_pos_l_f,
      class t_in_pos_r_f
   >
   struct natural_equi_join_t;

   /**
    * @brief Convenience function for Natural-Join-operator.
    * @details Joins the two given columns and returns the matching pairs of positions
    * in the form of two position columns of the same length.
    * Example:
    * - p_in_L_pos_column:  [ 22, 44, 11, 33, 55, 77 ]
    * - p_in_R_pos_column:  [ 33, 22, 22, 11 ]
    * - p_out_L_pos_column: [  0,  0,  2,  3 ]
    * - p_out_R_pos_column: [  1,  2,  3,  0 ]
    * Internally, a hash-map is used as an intermediate uncompressed datastructure.
    * The left column is used to build up the map and the right column is probed
    * against the map. Thus the used hash-map is optimized for vectorized processing,
    * it only supports arithmetic types for keys and values. Consequently, the left
    * column must consist of only unique values.
    *
    * @tparam t_vector_extension Used vector extension.
    * @tparam t_out_pos_l_f Compression format for the position list of the left result column.
    * @tparam t_out_pos_r_f Compression format for the position list of the right result column.
    * @tparam t_in_pos_l_f Compression format for the position list of the left input column.
    * @tparam t_in_pos_r_f Compression format for the position list of the right input column.
    * @param p_in_L_pos_column The left column to join on, containing only unique data elements.
    * @param p_in_R_pos_column The right column to join on.
    * @param p_out_count_estimate An optional estimate of the number of data
    * elements in the output column. If specified, the output column will allocate
    * enough memory for exactly this number of data elements. Otherwise, a
    * pessimistic estimation will be done.
    * @return A tuple of two columns containing equally many data elements. The
    * first (second) output column contains positions referring to the left
    * (right) input column. The i-th positions in the two output columns denote
    * one matching pair.
    */
   template<
      class t_vector_extension,
      class t_out_pos_l_f,
      class t_out_pos_r_f,
      class t_in_pos_l_f,
      class t_in_pos_r_f
   >
   std::tuple<
      column< t_out_pos_l_f > const *,
      column< t_out_pos_r_f > const *
   > const
   natural_equi_join(
      column< t_in_pos_l_f > const * const p_in_L_pos_column,
      column< t_in_pos_r_f > const * const p_in_R_pos_column,
      size_t const p_out_count_estimate = 0
   ) {
      return
         natural_equi_join_t<
            t_vector_extension,
            t_out_pos_l_f,
            t_out_pos_r_f,
            t_in_pos_l_f,
            t_in_pos_r_f
         >::apply( p_in_L_pos_column, p_in_R_pos_column, p_out_count_estimate );
   }

}

#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_JOIN_NATURAL_EQUI_H
