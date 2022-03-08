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
 * @file select_bm_uncompr.h
 * @brief Select-operator using scalar and bitmap processing style.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_SELECT_BM_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_SELECT_BM_UNCOMPR_H

#include <core/operators/interfaces/select_bm_pl.h>
#include <core/morphing/format.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/utils/math.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#include <cstdint>

namespace morphstore {

    /**
     * Select-operator. Outputs bitmap which are mapped to integer types (uint64_t) of all data elements
     * in the given column which fulfil the given predicate.
     *
     * Example:
     * - inDataCol: [95, 102, 100, 87, 120]
     * - predicate: "less than 100"
     * - bitmap:   [0 1 0 0 1] -> (int) 9
     * - outBmCol: [ 9 ]
     *
     * @param inDataCol The column to do the selection on.
     * @param val The constant each data element is compared to using the
     * comparison operation t_op.
     * @return A column containing the integer encoded bitmaps of all data elements d in
     * inDataCol for which t_op(d, val) is true.
     */

    // partial template specialization
    template<template<typename> class t_op>
    struct select_t<t_op,
            vectorlib::scalar<vectorlib::v64<uint64_t>>,
            bitmap_f<uncompr_f>,
            uncompr_f>
    {
        static
        const column< bitmap_f<uncompr_f> > *
        apply(
                const column<uncompr_f> * const inDataCol,
                const uint64_t val,
                const size_t outPosCountEstimate = 0
        ) {
            // to avoid unused parameter error -> we have to keep the outPosCountEstimate because of the declaration in interfaces/select.h
            (void)outPosCountEstimate;

            const size_t inDataCount = inDataCol->get_count_values();
            const uint64_t * const inData = inDataCol->get_data();

            // number of columns needed to represent bitmap in a list of 64-bit integers
            const size_t countBits = std::numeric_limits<uint64_t>::digits;
            const size_t outBmCountEstimate = round_up_div(inDataCount, countBits);

            auto outBmCol = new column< bitmap_f<uncompr_f> >
                    (outBmCountEstimate * sizeof(uint64_t));

            t_op<uint64_t> op;
            uint64_t * outBm = outBmCol->get_data();
            const uint64_t * const initOutBm = outBm;

            // bitmap word mapped to integer type
            uint64_t word = 0;

            // selection part
            for(unsigned i = 0; i < inDataCount; ++i){
                // store word if we reach end of current encoded bitmap, i.e. if i = 64 -> (i % 64) == 0
                // [ (i % n) == 0 ] <-> [ (i & n-1)  == 0 ] -> !(i & n-1)
                if(i && !(i & (countBits-1))){
                    *outBm = word;
                    outBm++;
                    word = 0;
                }

                // match
                if(op(inData[i], val)) {
                    // set bit at position i in current word, if countBits is power of 2 => (i & (countBits-1))
                    word |= 1ULL << (i & (countBits-1));
                }
            }

            // eventually store word in column (e.g. if we only process < 64 data points)
            if(word){
                *outBm = word;
                outBm++;
            }

            const size_t outBmCount = outBm - initOutBm;
            outBmCol->set_meta_data(outBmCount, outBmCount * sizeof(uint64_t));

            return outBmCol;
        }
    };
}

#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_SELECT_BM_UNCOMPR_H