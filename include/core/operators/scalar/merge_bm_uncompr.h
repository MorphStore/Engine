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
 * @file merge_bm_uncompr.h
 * @brief Template specialization of the merge-operator for uncompressed inputs and
 *        outputs using bitmaps as intermediate data structure + the scalar processing style.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_MERGE_BM_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_MERGE_BM_UNCOMPR_H

#include <core/operators/interfaces/merge_bm_pl.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#include <cstdint>

namespace morphstore {

    template <typename T>
    constexpr bool size_equal(const T a, const T b) {
        return a == b;
    }

    template<>
    struct merge_sorted_t<
            vectorlib::scalar<vectorlib::v64<uint64_t>>,
            bitmap_f<uncompr_f>,
            bitmap_f<uncompr_f>,
            bitmap_f<uncompr_f>
    >{
        static
        const column< bitmap_f<uncompr_f> > *
        apply(
                const column< bitmap_f<uncompr_f> > * const inBmLCol,
                const column< bitmap_f<uncompr_f> > * const inBmRCol
        ) {
            // differentiate the bitmaps according to their sizes
            const column< bitmap_f<uncompr_f> > * smallerBmCol;
            const column< bitmap_f<uncompr_f> > * largerBmCol;
            if(inBmLCol->get_size_used_byte() < inBmRCol->get_size_used_byte()) {
                smallerBmCol = inBmLCol;
                largerBmCol = inBmRCol;
            } else {
                smallerBmCol = inBmRCol;
                largerBmCol = inBmLCol;
            }

            const uint64_t * smallerBmPtr = smallerBmCol->get_data();
            const uint64_t * largerBmPtr = largerBmCol->get_data();
            const uint64_t * const endInBmSmaller = smallerBmPtr + smallerBmCol->get_count_values();
            const uint64_t * const endInBmLarger = largerBmPtr + largerBmCol->get_count_values();

            // if we have two different lengths of bitmaps, we allocate according to the larger one
            auto outBmCol = new column< bitmap_f<uncompr_f> >( largerBmCol->get_size_used_byte());

            uint64_t * outBm = outBmCol->get_data();
            const uint64_t * const initoutBm = outBm;

            // first we process according to the smaller bm's size
            while(smallerBmPtr < endInBmSmaller) {
                // logical OR-operation between the two bitmap encoded words
                *outBm = *smallerBmPtr | *largerBmPtr;
                outBm++;
                smallerBmPtr++;
                largerBmPtr++;
            }

            // finally, process the (smallest - largest) bits -> if there is any difference
            // just copy the bits in output bitmap
            // TODO: we could use memcpy() here to optimize this
            while(largerBmPtr < endInBmLarger){
                *outBm = *largerBmPtr;
                outBm++;
                largerBmPtr++;
            }

            const size_t outBmCount = outBm - initoutBm;
            outBmCol->set_meta_data(outBmCount, outBmCount * sizeof(uint64_t));

            return outBmCol;

        }
    };
}

#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_MERGE_BM_UNCOMPR_H