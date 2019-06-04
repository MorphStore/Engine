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
 * @file agg_sum_compr_iterator.h
 * @brief A whole-column aggregation operator on compressed data using the 
 * format's read iterator to access the compressed data.
 * @todo Currently, the iterator works only with scalars. When this changes, we
 * should move this file to another directory.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_COMPR_ITERATOR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_COMPR_ITERATOR_H

#include <core/operators/interfaces/agg_sum.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/processing_style.h>

#include <cstdint>

namespace morphstore {

template<processing_style_t t_ps, class t_in_data_f>
const column<uncompr_f> *
agg_sum(
        const column<t_in_data_f> * const inDataCol
) {
    const size_t inDataCount = inDataCol->get_count_values();
    
    // Exact allocation size (for uncompressed data).
    auto outDataCol = new column<uncompr_f>(sizeof(uint64_t));
    uint64_t * const outData = outDataCol->get_data();

    *outData = 0;
    read_iterator<t_in_data_f> it(inDataCol->get_data());
    for(unsigned i = 0; i < inDataCount; i++)
        *outData += it.next();
    
    outDataCol->set_meta_data(1, sizeof(uint64_t));
    
    return outDataCol;
}

}
#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_COMPR_ITERATOR_H
