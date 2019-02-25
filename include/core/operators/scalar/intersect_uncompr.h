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
 * @file intersect_uncompr.h
 * @brief Template specialization of the intersect-operator for uncompressed
 * inputs and outputs using the scalar processing style. Note that these are
 * simple reference implementations not tailored for efficiency.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_INTERSECT_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_INTERSECT_UNCOMPR_H

#include <core/operators/interfaces/intersect.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/processing_style.h>

#include <algorithm>
#include <cstdint>

namespace morphstore {
    
template<>
const column<uncompr_f> *
intersect_sorted<processing_style_t::scalar>(
        const column<uncompr_f> * const inPosLCol,
        const column<uncompr_f> * const inPosRCol,
        const size_t outPosCountEstimate
) {
    const uint64_t * inPosL = inPosLCol->get_data();
    const uint64_t * inPosR = inPosRCol->get_data();
    const uint64_t * const endInPosL = inPosL + inPosLCol->get_count_values();
    const uint64_t * const endInPosR = inPosR + inPosRCol->get_count_values();
    
    // If no estimate is provided: Pessimistic allocation size (for
    // uncompressed data), reached only if all positions in the smaller input
    // column are contained in the larger input column as well.
    auto outPosCol = new column<uncompr_f>(
            bool(outPosCountEstimate)
            // use given estimate
            ? (outPosCountEstimate * sizeof(uint64_t))
            // use pessimistic estimate
            : std::min(
                    inPosLCol->get_size_used_byte(),
                    inPosRCol->get_size_used_byte()
            )
    );
    uint64_t * outPos = outPosCol->get_data();
    const uint64_t * const initOutPos = outPos;
    
    while(inPosL < endInPosL && inPosR < endInPosR) {
        if(*inPosL < *inPosR)
            inPosL++;
        else if(*inPosR < *inPosL)
            inPosR++;
        else { // *inPosL == *inPosR
            *outPos = *inPosL;
            outPos++;
            inPosL++;
            inPosR++;
        }
    }
    
    const size_t outPosCount = outPos - initOutPos;
    outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
    
    return outPosCol;
}

}
#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_INTERSECT_UNCOMPR_H
