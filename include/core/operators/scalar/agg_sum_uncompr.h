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
 * @file agg_sum_uncompr.h
 * @brief Template specializations of the whole-column and group-related
 * variants of the aggregation(sum)-operator for uncompressed inputs and
 * outputs using the scalar processing style. Note that these are simple
 * reference implementations not tailored for efficiency.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_UNCOMPR_H

#include "../interfaces/agg_sum.h"
#include "../../morphing/format.h"
#include "../../storage/column.h"
#include "../../utils/basic_types.h"
#include "../../utils/processing_style.h"

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace morphstore {

template<>
const column<uncompr_f> *
agg_sum<processing_style_t::scalar>(
        const column<uncompr_f> * const inDataCol
) {
    const size_t inDataCount = inDataCol->get_count_values();
    const uint64_t * const inData = inDataCol->get_data();
    
    // Exact allocation size (for uncompressed data).
    auto outDataCol = new column<uncompr_f>(sizeof(uint64_t));
    uint64_t * const outData = outDataCol->get_data();

    for(unsigned i = 0; i < inDataCount; i++)
        *outData += inData[i];
    
    outDataCol->set_meta_data(1, sizeof(uint64_t));
    
    return outDataCol;
}

template<>
const column<uncompr_f> *
agg_sum<processing_style_t::scalar>(
        const column<uncompr_f> * const inGrCol,
        const column<uncompr_f> * const inDataCol,
        size_t inExtCount
) {
    const size_t inDataCount = inDataCol->get_count_values();
    
    if(inDataCount != inGrCol->get_count_values())
        throw std::runtime_error(
                "agg_sum: inGrCol and inDataCol must contain the same number "
                "of data elements"
        );
    
    const uint64_t * const inGr = inGrCol->get_data();
    const uint64_t * const inData = inDataCol->get_data();
    
    const size_t outDataSize = inExtCount * sizeof(uint64_t);
    // Exact allocation size (for uncompressed data).
    auto outDataCol = new column<uncompr_f>(outDataSize);
    uint64_t * const outData = outDataCol->get_data();
    
    for(unsigned i = 0; i < inDataCount; i++)
        outData[inGr[i]] += inData[i];
    
    outDataCol->set_meta_data(inExtCount, outDataSize);
    
    return outDataCol;
}
    
}
#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_UNCOMPR_H