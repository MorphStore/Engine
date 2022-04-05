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
 * @file write_iterator_IR_wah_compr_test.cpp
 * @brief Special test for WAH-format's write_iterator_IR implementation which is used to compress bitmaps.

 *
 *        For more details, see sequential write section in core/morphing/wah.h
 *
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/printing.h>
#include <core/utils/basic_types.h>

#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>

#include <core/operators/general_vectorized/select_pl_compr.h>
#include <core/operators/general_vectorized/select_bm_uncompr.h>
#include <core/morphing/intermediates/transformations/transformation_algorithms.h>

#include <core/morphing/wah.h>
#include <core/morphing/uncompr.h>

#include <iostream>

    #define TEST_DATA_COUNT 1000*1500

using namespace morphstore;
using namespace vectorlib;

int main(void) {

    /**         PoC for WAH - write_iterator_IR:
     *                 (1) Generate a uncompressed bitmap data
     *                 (2) Compressed position-list select-operator that outputs a WAH-compressed bitmap (includes IR-transformation)
     *                 (3) Uncompressed bitmap select-operator with same predicate evaluation as (2)
     *                 (4) Decompression of (2): WAH-bitmap to uncompressed-bitmap
     *                 (5) Verify results: (3) == (4)
     */

    const uint64_t predicate = 33333;

    // (1) Generate uncompressed base data
    std::cout << "Generating..." << std::flush;
    auto inCol = generate_with_distr(
            TEST_DATA_COUNT,
            std::uniform_int_distribution<uint64_t>(
                    1,
                    TEST_DATA_COUNT - 1
            ),
            false
    );
    std::cout << "Done...\n";

    // (2) position-list SELECT-operator with compressed-WAH-bitmap as output
    auto result_compr =
            select_pl_wit_t<
                greater,
                avx2<v256<uint64_t>>,
                bitmap_f<wah_f>,
                uncompr_f
            >::apply(inCol, predicate);

    // (3) additional uncompressed result to verify output later on
    auto result_uncompr=
            morphstore::select<
                greater,
                avx2<v256<uint64_t>>,
                bitmap_f<uncompr_f>,
                uncompr_f
            >(inCol, predicate);

    // (4) decompress WAH-bitmap
    auto bm_decompr =
            morph_t<
                avx2<v256<uint64_t>>,
                bitmap_f<uncompr_f>,
                bitmap_f<wah_f>
            >::apply(result_compr);

    // (5) Validate decompressed == uncompressed ?
    const bool allGood =
            memcmp(result_uncompr->get_data(), bm_decompr->get_data(), (int)(result_uncompr->get_count_values()*8)); //returns zero if all bytes match

    return allGood;
}