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
 * @file join_uncompr.h
 * @brief Template specialization of the join-operators for uncompressed inputs
 * and outputs using the scalar processing style. Note that these are simple
 * reference implementations not tailored for efficiency.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_JOIN_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_JOIN_UNCOMPR_H

#include <core/operators/interfaces/join.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#include <cstdint>
#include <tuple>

namespace morphstore {
    
template<>
const std::tuple<
        const column<uncompr_f> *,
        const column<uncompr_f> *
>
nested_loop_join<vector::scalar<vector::v64<uint64_t>>>(
        const column<uncompr_f> * const inDataLCol,
        const column<uncompr_f> * const inDataRCol,
        const size_t outCountEstimate
) {
    const size_t inDataLCount = inDataLCol->get_count_values();
    const size_t inDataRCount = inDataRCol->get_count_values();
    
    // Ensure that the left column is the larger one, swap the input and output
    // column order if necessary.
    if(inDataLCount < inDataRCount) {
        auto outPosRL = nested_loop_join<
                vector::scalar<vector::v64<uint64_t>>,
                uncompr_f,
                uncompr_f
        >(
                inDataRCol,
                inDataLCol,
                outCountEstimate
        );
        return std::make_tuple(std::get<1>(outPosRL), std::get<0>(outPosRL));
    }
    
    const uint64_t * const inDataL = inDataLCol->get_data();
    const uint64_t * const inDataR = inDataRCol->get_data();
    
    // If no estimate is provided: Pessimistic allocation size (for
    // uncompressed data), reached only if the result is the cross product of
    // the two input columns.
    const size_t size = bool(outCountEstimate)
            // use given estimate
            ? (outCountEstimate * sizeof(uint64_t))
            // use pessimistic estimate
            : (inDataLCount * inDataRCount * sizeof(uint64_t));
    auto outPosLCol = new column<uncompr_f>(size);
    auto outPosRCol = new column<uncompr_f>(size);
    uint64_t * const outPosL = outPosLCol->get_data();
    uint64_t * const outPosR = outPosRCol->get_data();
    
    unsigned iOut = 0;
    for(unsigned iL = 0; iL < inDataLCount; iL++)
        for(unsigned iR = 0; iR < inDataRCount; iR++)
            if(inDataL[iL] == inDataR[iR]) {
                outPosL[iOut] = iL;
                outPosR[iOut] = iR;
                iOut++;
            }
    
    const size_t outSize = iOut * sizeof(uint64_t);
    outPosLCol->set_meta_data(iOut, outSize);
    outPosRCol->set_meta_data(iOut, outSize);
    
    return std::make_tuple(outPosLCol, outPosRCol);
}

template<>
const column<uncompr_f> *
left_semi_nto1_nested_loop_join<vector::scalar<vector::v64<uint64_t>>>(
        const column<uncompr_f> * const inDataLCol,
        const column<uncompr_f> * const inDataRCol,
        const size_t outCountEstimate
) {
    const size_t inDataLCount = inDataLCol->get_count_values();
    const size_t inDataRCount = inDataRCol->get_count_values();
    
    const uint64_t * const inDataL = inDataLCol->get_data();
    const uint64_t * const inDataR = inDataRCol->get_data();
    
    // If no estimate is provided: Pessimistic allocation size (for
    // uncompressed data), reached only if all data elements in the left input
    // column have a join partner in the right input column.
    const size_t size = bool(outCountEstimate)
            // use given estimate
            ? (outCountEstimate * sizeof(uint64_t))
            // use pessimistic estimate
            : (inDataLCount * sizeof(uint64_t));
    auto outPosLCol = new column<uncompr_f>(size);
    uint64_t * const outPosL = outPosLCol->get_data();
    
    unsigned iOut = 0;
    for(unsigned iL = 0; iL < inDataLCount; iL++)
        for(unsigned iR = 0; iR < inDataRCount; iR++)
            if(inDataL[iL] == inDataR[iR]) {
                outPosL[iOut] = iL;
                iOut++;
                break;
            }
    
    outPosLCol->set_meta_data(iOut, iOut * sizeof(uint64_t));
    
    return outPosLCol;
}

}
#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_JOIN_UNCOMPR_H
