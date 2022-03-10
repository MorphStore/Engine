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
 * @file bm_morph_test.cpp
 * @brief Small test of compression/decompression of bitmaps.
 *
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/printing.h>
#include <core/morphing/format.h>
#include <core/morphing/rle.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/morphing/intermediates/bitmap.h>

#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>

#include <iostream>
#include <type_traits>

#define TEST_DATA_COUNT 1000

using namespace morphstore;
using namespace vectorlib;

int main(void) {


    // ************************************************************************
    // * Morph-Operations
    // ************************************************************************

    /**
     * General idea: 1.    uncompr_f              ->  rle_f                          (compression)
     *               1.1   bitmap_f<uncompr_f>   ->  bitmap_f<rle_f>               (compression)
     *               2.    rle_f                  ->  uncompr_f                      (decompression)
     *               2.2   bitmap_f<rle_f>       ->  bitmap_f<uncompr_f>           (decompression)
     *               3.    compare results for equality
     */
    using compr_f = rle_f;

    // ***** 1. & 1.1 -- Compression *****
    auto uncompr_1 = generate_with_distr(
            TEST_DATA_COUNT,
            std::uniform_int_distribution<uint64_t>(1, TEST_DATA_COUNT),
            false
    );
    auto bm_uncompr_1 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(uncompr_1);
    //print_columns(print_buffer_base::decimal, pl_uncompr_1, "pl-uncompressed-1");

    auto compr_1 =
            morph_t<
                avx2<v256<uint64_t>>,
                compr_f,
                uncompr_f
            >::apply(uncompr_1);

    auto bm_compr_1 =
            morph_t<
                avx2<v256<uint64_t>>,
                bitmap_f<compr_f>,
                bitmap_f<uncompr_f>
            >::apply(bm_uncompr_1);

    // ***** 2. && 2.2 -- Decompression *****
    auto uncompr_2 =
            morph_t<
                avx2<v256<uint64_t>>,
                uncompr_f,
                compr_f
            >::apply(compr_1);

    auto bm_uncompr_2 =
            morph_t<
                avx2<v256<uint64_t>>,
                bitmap_f<uncompr_f>,
                bitmap_f<compr_f>
            >::apply(bm_compr_1);

    // ***** 3. compare results*****
    const bool allGood_1 =
            memcmp(compr_1->get_data(), bm_compr_1->get_data(), (int)(TEST_DATA_COUNT*8));

    const bool allGood_2 =
            memcmp(uncompr_2->get_data(), bm_uncompr_2->get_data(), (int)(TEST_DATA_COUNT*8));

    const bool allGood_3 =
            memcmp(bm_uncompr_1->get_data(), bm_uncompr_2->get_data(), (int)(TEST_DATA_COUNT*8));

    const bool allGood_4 =
            memcmp(uncompr_1->get_data(), uncompr_2->get_data(), (int)(TEST_DATA_COUNT*8));

    return allGood_1 || allGood_2 || allGood_3 || allGood_4;
}