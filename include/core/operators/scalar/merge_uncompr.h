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
 * @file merge_uncompr.h
 * @brief Template specialization of the merge-operator for uncompressed inputs
 * and outputs using the scalar processing style. Note that these are simple
 * reference implementations not tailored for efficiency.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_MERGE_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_MERGE_UNCOMPR_H

#include <core/operators/interfaces/merge.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#include <cstdint>

namespace morphstore {
    
template<>
const column<uncompr_f> *
merge_sorted<vector::scalar<vector::v64<uint64_t>>>(
        const column<uncompr_f> * const inPosLCol,
        const column<uncompr_f> * const inPosRCol,
        const size_t outPosCountEstimate
) {
    const uint64_t * inPosL = inPosLCol->get_data();
    const uint64_t * inPosR = inPosRCol->get_data();
    const uint64_t * const inPosLEnd = inPosL + inPosLCol->get_count_values();
    const uint64_t * const inPosREnd = inPosR + inPosRCol->get_count_values();
    
    // If no estimate is provided: Pessimistic allocation size (for
    // uncompressed data), reached only if the two input columns are disjoint.
    auto outPosCol = new column<uncompr_f>(
            bool(outPosCountEstimate)
            // use given estimate
            ? (outPosCountEstimate * sizeof(uint64_t))
            // use pessimistic estimate
            : (
                    inPosLCol->get_size_used_byte() +
                    inPosRCol->get_size_used_byte()
            )
    );
    uint64_t * outPos = outPosCol->get_data();
    const uint64_t * const initOutPos = outPos;
    
    while(inPosL < inPosLEnd && inPosR < inPosREnd) {
        if(*inPosL < *inPosR) {
            *outPos = *inPosL;
            inPosL++;
        }
        else if(*inPosR < *inPosL) {
            *outPos = *inPosR;
            inPosR++;
        }
        else { // *inPosL == *inPosR
            *outPos = *inPosL;
            inPosL++;
            inPosR++;
        }
        outPos++;
    }
    // At this point, at least one of the operands has been fully consumed and
    // the other one might still contain data elements, which must be output.
    while(inPosL < inPosLEnd) {
        *outPos = *inPosL;
        outPos++;
        inPosL++;
    }
    while(inPosR < inPosREnd) {
        *outPos = *inPosR;
        outPos++;
        inPosR++;
    }
    
    const size_t outPosCount = outPos - initOutPos;
    outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
    
    return outPosCol;
}

}
#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_MERGE_UNCOMPR_H
