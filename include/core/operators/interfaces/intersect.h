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
 * @file intersect.h
 * @brief The template-based interface of the intersect-operator.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_INTERSECT_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_INTERSECT_H

#include <core/storage/column.h>
#include <core/utils/basic_types.h>

namespace morphstore {
    
/**
 * Intersect-operator for sorted inputs. Intersects the two given columns, each
 * of which is assumed to be sorted in ascending order and to contain only
 * unique data elements. This operator is commutative, i.e., interchanging its
 * first and second operand does not affect the result.
 * 
 * Example:
 * - inPosLCol: [1, 4, 5,    8, 9, 12    ]
 * - inPosRCol: [1,       6, 8,    12, 15]
 * - outPosCol: [1,          8,    12    ]
 * 
 * @param inPosLCol A column of sorted and unique data elements.
 * @param inPosRCol A column of sorted and unique data elements.
 * @param outPosCountEstimate An optional estimate of the number of data
 * elements in the output position column. If specified, the output positions
 * column will allocate enough memory for exactly this number of data elements.
 * Otherwise, a pessimistic estimation will be done.
 * @return The intersection of the two input columns, which is also sorted and
 * unique.
 */
template<
        class t_vector_extension,
        class t_out_pos_f,
        class t_in_pos_l_f,
        class t_in_pos_r_f
>
const column<t_out_pos_f> *
intersect_sorted(
        const column<t_in_pos_l_f> * const inPosLCol,
        const column<t_in_pos_r_f> * const inPosRCol,
        const size_t outPosCountEstimate = 0
);

}
#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_INTERSECT_H
