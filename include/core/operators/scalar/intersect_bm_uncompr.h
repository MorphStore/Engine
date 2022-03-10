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
 * @file intersect_bm_uncompr.h
 * @brief Template specialization of the intersect-operator for uncompressed inputs and
 *        outputs using bitmaps as intermediate data structure + the scalar processing style.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_INTERSECT_BM_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_INTERSECT_BM_UNCOMPR_H

#include <core/operators/interfaces/intersect_bm_pl.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#include <algorithm>
#include <cstdint>

namespace morphstore {

    template<>
    const column< bitmap_f<uncompr_f> > *
    intersect_sorted<
            vectorlib::scalar<vectorlib::v64<uint64_t>>,
            bitmap_f<uncompr_f>,
            bitmap_f<uncompr_f>,
            bitmap_f<uncompr_f>
    >(
        const column< bitmap_f<uncompr_f> > * const inBmLCol,
        const column< bitmap_f<uncompr_f> > * const inBmRCol,
        const size_t outBmCountEstimate
    ) {
        const uint64_t * inBmL = inBmLCol->get_data();
        const uint64_t * inBmR = inBmRCol->get_data();
        const uint64_t * const endInBmL = inBmL + inBmLCol->get_count_values();
        const uint64_t * const endInBmR = inBmR + inBmRCol->get_count_values();

        // if no estimation is given, use the smaller input bitmap size
        auto outBmCol = new column< bitmap_f<uncompr_f> >(
                bool(outBmCountEstimate)
                // use given estimate
                ? (outBmCountEstimate * sizeof(uint64_t))
                // use pessimistic estimate
                : std::min(
                        inBmLCol->get_size_used_byte(),
                        inBmRCol->get_size_used_byte()
                )
        );

        uint64_t * outBm = outBmCol->get_data();
        const uint64_t * const initoutBm = outBm;

        while(inBmL < endInBmL && inBmR < endInBmR) {
            // logical AND-operation between the two bitmap encoded words
            *outBm = *inBmL & *inBmR;
            outBm++;
            inBmL++;
            inBmR++;
        }

        const size_t outBmCount = outBm - initoutBm;
        outBmCol->set_meta_data(outBmCount, outBmCount * sizeof(uint64_t));

        return outBmCol;
    }
}

#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_INTERSECT_BM_UNCOMPR_H
