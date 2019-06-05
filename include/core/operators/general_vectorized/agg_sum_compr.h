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
 * @file agg_sum_compr.h
 * @brief Whole-column aggregation-operator based on the vector-lib, weaving
 * the operator's core into the decompression routine of the input data's
 * format.
 * @todo Currently, it is not truly general-vectorized, because the current
 * implementation of decompress_and_process_batch is hand-written scalar code.
 * @todo Support columns of arbitrary length w.r.t. vectorization and
 * compression.
 */

#ifndef AGG_SUM_COMPR_H
#define AGG_SUM_COMPR_H

#include <core/morphing/format.h>
#include <core/operators/general_vectorized/agg_sum_uncompr.h>
#include <core/operators/interfaces/agg_sum.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>

#include <cstdint>

namespace morphstore {

    template<
            class t_vector_extension,
            class t_in_data_f
    >
    const column<uncompr_f> *
    agg_sum(
            const column<t_in_data_f> * const inDataCol
    ) {
        const uint8_t * inData = inDataCol->get_data();
        
        typename agg_sum_processing_unit<t_vector_extension>::state_t s;
        decompress_and_process_batch<
                t_vector_extension, t_in_data_f, agg_sum_processing_unit
        >::apply(
            inData, inDataCol->get_size_used_byte(), s 
        );
        
        auto outDataCol = new column<uncompr_f>(sizeof(uint64_t));
        uint64_t * const outData = outDataCol->get_data();
        *outData = agg_sum_processing_unit<t_vector_extension>::finalize(s);
        outDataCol->set_meta_data(1, sizeof(uint64_t));
        
        return outDataCol;
    }

}

#endif /* AGG_SUM_COMPR_H */

