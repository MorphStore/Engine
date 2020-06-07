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

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_SELECT_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_SELECT_H

#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <cstdint>

namespace morphstore {
    
/**
 * A struct wrapping the actual select-operator. This is necessary to enable
 * partial template specialization.
 */
template<
        template<class, int> class t_compare,
        //template<typename> class t_op,
        class t_vector_extension,
        class t_out_pos_f,
        class t_in_data_f
>
struct select_t {
    /**
     * Select-operator. Outputs the positions of all data elements in the given
     * column which fulfil the given predicate.
     * 
     * Example:
     * - inDataCol: [95, 102, 100, 87, 120]
     * - predicate: "less than 100"
     * - outPosCol: [ 0,            3     ]
     * 
     * @param inDataCol The column to do the selection on.
     * @param val The constant each data element is compared to using the
     * comparison operation t_op.
     * @param outPosCountEstimate An optional estimate of the number of data
     * elements in the output position column. If specified, the output
     * positions column will allocate enough memory for exactly this number of
     * data elements. Otherwise, a pessimistic estimation will be done.
     * @return A column containing the positions of all data elements d in
     * inDataCol for which t_op(d, val) is true.
     */
    static
    const column<t_out_pos_f> *
    apply(
        const column<t_in_data_f> * const inDataCol,
        const uint64_t val,
        const size_t outPosCountEstimate = 0
    );
};

template<
        template<class, int> class t_compare,
        //template<typename> class t_op,
        class t_vector_extension,
        class t_out_pos_f,
        class t_in_data_f
>
    static
    const column<t_out_pos_f> *
select(
        const column<t_in_data_f> * const inDataCol,
        const uint64_t val,
        const size_t outPosCountEstimate = 0
) {
    return select_t<t_compare, //t_op,
        t_vector_extension,
        t_out_pos_f,
        t_in_data_f>::apply(
            inDataCol,
            val,
            outPosCountEstimate
    );
}


}
#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_SELECT_H
