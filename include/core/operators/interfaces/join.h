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
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_JOIN_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_JOIN_H

#include "../../storage/column.h"
#include "../../utils/processing_style.h"

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
 * @return A tuple of two columns containing equally many data elements. The
 * first (second) output column contains positions referring to the left
 * (right) input column. The i-th positions in the two output columns denote
 * one matching pair.
 */
template<
        processing_style_t t_ps,
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
        const column<t_in_data_r_f> * const inDataRCol
);

}
#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_JOIN_H