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
 * @file select.h
 * @brief The template-based interface of the select-operator.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_BETWEEN_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_BETWEEN_H

#include <core/storage/column.h>
#include <core/utils/basic_types.h>

#include <cstdint>

namespace morphstore {

/**
 * A struct wrapping the actual select-operator. This is necessary to enable
 * partial template specialization.
 */
   template<
      template<class, int> class t_op_lower,
      template<class, int> class t_op_upper,
      class t_vector_extension,
      class t_out_pos_f,
      class t_in_data_f
   >
   struct between_t;
      /**
       * Select-operator. Outputs the positions of all data elements in the given
       * column which fulfil the given predicate.
       *
       * Example:
       * - inDataCol: [95, 102, 100, 87, 120]
       * - predicate: "greater than 95, less than 110"
       * - outPosCol: [   1,  2,         ]
       * @tparam t_vector_extension The column to do the selection on.
       * @tparam t_op_lower The operator to compare with the lower bound, typically greater or greaterequal
       * @tparam t_op_upper The operator to compare with the upper bound, typically less or lessequal
       * @tparam t_out_pos_f The format of the returned column
       * @tparam t_in_data_f The format of the input column inDataCol
       * @param inDataCol The column to do the selection on.
       * @param val_lower The lower bound each data element is compared to using the
       * comparison operation t_op.
       * @param val_upper The upper bound each data element is compared to using the
       * comparison operation t_op.
       * @param outPosCountEstimate An optional estimate of the number of data
       * elements in the output position column. If specified, the output
       * positions column will allocate enough memory for exactly this number of
       * data elements. Otherwise, a pessimistic estimation will be done.
       * @return A column containing the positions of all data elements d in
       * inDataCol for which t_op(d, val) is true.
       */
      

   template<
      template<class, int> class t_op_lower,
      template<class, int> class t_op_upper,
      class t_vector_extension,
      class t_out_pos_f,
      class t_in_data_f
   >
   static
   const column<t_out_pos_f> *
   between(
      column<t_in_data_f> const * const inDataCol,
      uint64_t const val_lower,
      uint64_t const val_upper,
      size_t const outPosCountEstimate = 0
   ) {
      return
         between_t<t_op_lower, t_op_upper,
            t_vector_extension,
            t_out_pos_f,
            t_in_data_f
         >::apply(
            inDataCol,
            val_lower,
            val_upper,
            outPosCountEstimate
         );
   }


}
#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_BETWEEN_H
