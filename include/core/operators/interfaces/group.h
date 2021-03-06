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
 * @file group.h
 * @brief The template-based interfaces of the unary and binary
 * group-operators.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_GROUP_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_GROUP_H

#include <core/storage/column.h>
#include <core/utils/basic_types.h>

#include <tuple>

namespace morphstore {
    
/**
 * Unary group-operator. Used for grouping by a single column or as the initial
 * step for grouping by multiple columns (the subsequent steps have to be done
 * using the binary group-operator).
 * 
 * Example:
 * - inDataCol: [333, 333, 111, 333, 222, 111]
 * - outGrCol:  [  0,   0,   1,   0,   2,   1]
 * - outExtCol: [  0,        2,        4     ]
 * 
 * @param inDataCol The column to group by. Each distinct value in this column
 * yields one group in the output.
 * @param outExtCountEstimate An optional estimate of the number of data
 * elements in the output extents column. If specified, the output extents
 * column will allocate enough memory for exactly this number of data elements.
 * Otherwise, a pessimistic estimation will be done.
 * @return A tuple of two columns. The first one contains as many data elements
 * as inDataCol and its i-th data element is the group-id of the i-th data
 * element in inDataCol. The group-ids are positions referring to the second
 * output column, the so-called extents. This second column contains as many
 * data elements as there are groups and its i-th data element is the position
 * of some representative of the i-th group in inDataCol.
 */
template<
        class t_vector_extension,
        class t_out_gr_f,
        class t_out_ext_f,
        class t_in_data_f
        
>
const std::tuple<
        const column<t_out_gr_f> *,
        const column<t_out_ext_f> *
>
group(
        const column<t_in_data_f> * const inDataCol,
        const size_t outExtCountEstimate = 0
);

/**
 * Binary group-operator. Used as a follow-up step for grouping by multiple
 * columns. Refines an existing grouping by taking an additional column into
 * account.
 * 
 * Example:
 * - inGrCol:    [ 0,  0,  1,  0,  2,  1]
 * - inDataCol:  [77, 88, 77, 77, 88, 88]
 * - outGrCol:   [ 0,  1,  2,  0,  3,  4]
 * - outExtCol:  [ 0,  1,  2,      4,  5]
 * 
 * @param inGrCol A column of group-ids obtained by a previous grouping step.
 * Must contain as many data elements as inDataCol.
 * @param inDataCol The additional column to group by. Each distinct
 * combination of a value and a corresponding group-id in inGrCol yields one
 * group in the output. Must contain as many data elements as inGrCol.
 * @return A tuple of two columns. The first one contains as many data elements
 * as inDataCol and its i-th data element is the group-id of the i-th data
 * element in inDataCol. The group-ids are positions referring to the second
 * output column, the so-called extents. This second column contains as many
 * data elements as there are groups and its i-th data element is the position
 * of some representative of the i-th group in inDataCol.
 */
template<
        class t_vector_extension,
        class t_out_gr_f,
        class t_out_ext_f,
        class t_in_gr_f,
        class t_in_data_f
>
const std::tuple<
        const column<t_out_gr_f> *,
        const column<t_out_ext_f> *
>
group(
        const column<t_in_gr_f> * const inGrCol,
        const column<t_in_data_f> * const inDataCol,
        const size_t outExtCountEstimate = 0
);

}
#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_GROUP_H
