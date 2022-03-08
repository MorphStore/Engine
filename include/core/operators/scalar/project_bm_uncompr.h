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
 * @file project_bm_uncompr.h
 * @brief Project-operator using scalar and bitmap processing style.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_PROJECT_BM_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_PROJECT_BM_UNCOMPR_H

#include <core/operators/interfaces/project_bm_pl.h>
#include <core/morphing/format.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#include <cstdint>

namespace morphstore {

    /**
     * @brief Project-operator. Extracts the data elements at the given
     * positions from the given data column using bitmaps.
     *
     * Example:
     * - `inDataCol`:  `[11, 44, 22, 33, 11]`
     * - `inBmCol`:   `[ 26 ]` -> bitmap: 11010 => positions: 1,3,4
     * - `outDataCol`: `[    44,     33, 11]`
     *
     * @param inDataCol The column to extract from.
     * @param inBmCol The column containing the encoded bitmaps with the positions to extract.
     * @return A column containing the extracted data elements, i.e., as many
     * as in `inBmCol`.
     */

    template<>
    struct project_t<
            vectorlib::scalar<vectorlib::v64<uint64_t>>,
            uncompr_f,
            uncompr_f,
            bitmap_f<uncompr_f>
    > {
        static
        const column<uncompr_f> *
        apply(
                const column<uncompr_f> * const inDataCol,
                const column< bitmap_f<uncompr_f> > * const inBmCol
        ) {
            const uint64_t * const inData = inDataCol->get_data();
            const uint64_t * const inBm = inBmCol->get_data();
            const size_t inBmCount = inBmCol->get_count_values();
            const size_t wordBitSize = sizeof(uint64_t) << 3;

            // @todo: Think about how to efficiently allocate the size for output column.
            //        We do not (yet) know how many bits are set in all encoded bitmaps in advance.
            //        Naive solution: iterate through all encoded bitmaps and count set bits using ::count_matches()
            //                        OR store in metadata the number of matches
            //      => For now, we use pessimistic allocation, i.e. #bitmaps * 64 * sizeof(uint64_t), i.e. every bit valid
            auto outDataCol = new column<uncompr_f>(
                    inBmCount * wordBitSize * sizeof(uint64_t)
            );
            uint64_t * outData = outDataCol->get_data();
            const uint64_t * const initOutData = outData;

            for(unsigned i = 0; i < inBmCol->get_count_values(); ++i) {
                // get current word, i.e. integer encoded bitmap
                uint64_t word = inBm[i];

                const size_t offset = i * wordBitSize;
                uint64_t trailingZeros, pos = 0ULL;

                while(word) {
                    // count trailing zeros
                    trailingZeros = __builtin_ctzll(word);
                    pos += trailingZeros;

                    // project value from inData using pos from bitmap
                    *outData = inData[pos + offset];

                    outData++;
                    pos++;
                    trailingZeros++;

                    // shift trailingZeros-bits out of the current word
                    // shifting it by more than or equal to 64 results in undefined behavior -> manually setting it to 0
                    // e.g. when uint64_t n = 2^63 -> binary = 10.......00 -> trailingZeros = 63, gets incremented then 64
                    word = (trailingZeros ^ wordBitSize) ?
                           word >> trailingZeros :
                           0;
                }
            }

            const size_t outDataCount = outData - initOutData;
            outDataCol->set_meta_data(outDataCount, outDataCount * sizeof(uint64_t));

            return outDataCol;
        }
    };

}

#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_PROJECT_BM_UNCOMPR_H