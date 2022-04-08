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
 * @file wah_bm_test.cpp
 * @brief Tests of the (de)compression morph operators for `wah_f`, Word-Aligned Hybrid compression for bitmaps.
 *
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/wah.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/utils/printing.h>

#include <core/operators/general_vectorized/select_bm_uncompr.h>

#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>

#include <iostream>

#define TEST_DATA_COUNT 100*100*10

using namespace morphstore;
using namespace vectorlib;

int main(void) {

    /** General idea:
     *                 (1) Generate a uncompressed bitmap
     *                 (2) morph: uncompressed-bm to wah_f (compression)
     *                 (3) morph: wah_f-compressed-bm to uncompr_f (decompression)
     *                 (4) compare results 1 + 3 (equality)
     */
    // ------------------------------------------------------------------------
    // (1) Uncompressed Bitmap-Data Generation
    // ------------------------------------------------------------------------

    /*
    // simple custom bitmap-encoded words
    auto bm_uncompr = reinterpret_cast<const column< bitmap_f<uncompr_f> > *>(
            //make_column({0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0,0, 123})
            //make_column({0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF})
            //make_column({0, 0, 0, 0})
            //make_column({1, 2, 3, 4, 5})
            //make_column({0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF})
    );
     */

    // get bitmap from select-operator (easier to generate a large bitmap)
    std::cout << "Generating..." << std::flush;
    auto inCol = generate_with_distr(
            TEST_DATA_COUNT,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    TEST_DATA_COUNT - 1
            ),
            false
    );
    std::cout << "Done...\n";

    // random selection: smaller than ...
    const uint64_t predicate = 1000;
    auto bm_uncompr =
            morphstore::select<
                less,
                avx2<v256<uint64_t>>,
                bitmap_f<uncompr_f>,
                uncompr_f
            >(inCol, predicate);

    //print_columns(print_buffer_base::decimal, bm_uncompr, "bm_uncompr");

    // ------------------------------------------------------------------------
    // (2) Compression
    // ------------------------------------------------------------------------

    // compress bitmap
    auto bm_compr =
            morph_t<
                avx2<v256<uint64_t>>,
                bitmap_f<wah_f>,
                bitmap_f<uncompr_f>
            >::apply(bm_uncompr);

    //print_columns(print_buffer_base::decimal, bm_compr, "bm_compr");

    // ------------------------------------------------------------------------
    // (3) Decompression
    // ------------------------------------------------------------------------

    // decompress WAH-bitmap
    auto bm_decompr =
            morph_t<
                avx2<v256<uint64_t>>,
                bitmap_f<uncompr_f>,
                bitmap_f<wah_f>
            >::apply(bm_compr);

    //print_columns(print_buffer_base::decimal, bm_decompr, "bm_decompr");

    // ------------------------------------------------------------------------
    // (4) Validation
    // ------------------------------------------------------------------------

    // check if bm_uncompr == bm_decompr
    const bool allGood =
            memcmp(bm_uncompr->get_data(), bm_decompr->get_data(), (int)(bm_uncompr->get_count_values()*8)); //returns zero if all bytes match

    return allGood;
}