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
 * @file calc.h
 * @brief The template-based interface of the unary and binary
 * calculation-operator.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_CALC_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_CALC_H

#include <core/storage/column.h>
#include <core/utils/basic_types.h>

namespace morphstore {
    
/**
 * A struct wrapping the actual unary calculation-operator. This is necessary
 * to enable partial template specialization.
 */
template<
        template<class, int> class t_unary_op,
        //template<typename> class t_unary_op,
        class t_vector_extension,
        class t_out_data_f,
        class t_in_data_f
>
struct calc_unary_t {
    /**
     * Unary calculation-operator. Applies the unary operation t_unary_op to
     * each data element in the given column.
     * 
     * Example:
     * - inDataCol:  [10, 20, 0, 3, 100]
     * - operation: "+5"
     * - outDataCol: [15, 25, 5, 8, 105]
     * 
     * @param inDataCol The column containing the operands.
     * @return A column containing as many data elements as the input column,
     * whereby the i-th data element is the result of the operation t_unary_op
     * applied to to the i-th data element of input column.
     */
    // static
    // const column<t_out_data_f> *
    // apply(const column<t_in_data_f> * const inDataCol);
        static
        const column<t_out_data_f> *
        apply(
            const column<t_in_data_f> * const inDataCol,
            const uint64_t val,
            const size_t outPosCountEstimate = 0
        );    
};
    
/**
 * A struct wrapping the actual binary calculation-operator. This is necessary
 * to enable partial template specialization.
 */
template<
        template<typename> class t_binary_op,
        class t_vector_extension,
        class t_out_data_f,
        class t_in_data_l_f,
        class t_in_data_r_f
>
struct calc_binary_t {
    /**
     * Binary calculation-operator. Applies the binary operation t_binary_op to
     * each pair of corresponding data elements in the two given data columns.
     * 
     * Example:
     * - inDataLCol: [10, 20, 0, 3, 100]
     * - inDataRCol: [ 5, 33, 3, 4,  10]
     * - operation: "+"
     * - outDataCol: [15, 53, 3, 7, 110]
     * 
     * @param inDataLCol The column containing the left operands. Must contain
     * as many data elements as inDataRCol.
     * @param inDataRCol The column containing the right operands. Must contain
     * as many data elements as inDataLCol.
     * @return A column containing as many data elements as each of the input
     * columns, whereby the i-th data element is the result of the operation
     * t_binary_op applied to the i-th data elements of the two input columns.
     */
    static
    const column<t_out_data_f> *
    apply(
            const column<t_in_data_l_f> * const inDataLCol,
            const column<t_in_data_r_f> * const inDataRCol
    );
};
 

template<
        template<typename> class t_binary_op,
        class t_vector_extension,
        class t_out_data_f,
        class t_in_data_l_f,
        class t_in_data_r_f
>

const column<t_out_data_f> *
    calc_binary(const column<t_in_data_l_f> * const inDataLCol,
            const column<t_in_data_r_f> * const inDataRCol){


    return calc_binary_t<t_binary_op,t_vector_extension,t_out_data_f,t_in_data_l_f,t_in_data_r_f>::apply(
            inDataLCol,
            inDataRCol
    );
}

template<
        template<class, int> class t_unary_op,
        //template<typename> class t_binary_op,
        class t_vector_extension,
        class t_out_data_f,
        class t_in_data_f
>

const column<t_out_data_f> *
    calc_unary(
        const column<t_in_data_f> * const inDataLCol,
        const uint64_t val,
        const size_t outPosCountEstimate = 0
    ){


    return calc_unary_t<t_unary_op,t_vector_extension,t_out_data_f,t_in_data_f>::apply(
            inDataLCol,
            val,
            outPosCountEstimate
    );

// const column<t_out_data_f> *
//     calc_unary(const column<t_in_data_f> * const inDataLCol){


//     return calc_unary_t<t_binary_op,t_vector_extension,t_out_data_f,t_in_data_f>::apply(
//             inDataLCol
//     );    
}

}
#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_CALC_H
