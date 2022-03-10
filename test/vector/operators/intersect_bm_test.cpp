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
 * @file intersect_bm_test.h
 * @brief Small test for vectorized intersect-operator using bitmaps.
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>

#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>
#include <vector/simd/avx2/primitives/manipulate_avx2.h>

#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include <vector/simd/sse/primitives/compare_sse.h>
#include <vector/simd/sse/primitives/manipulate_sse.h>

#include <core/operators/general_vectorized/intersect_bm_uncompr.h>

#include <core/utils/printing.h>

#define TEST_DATA_COUNT 100

int main( void ) {
    using namespace morphstore;
    using namespace vectorlib;

    /**
     *  @brief: Testing vectorized intersect-operator for bitmaps to enable execution of multidimensional boolean queries.
     *          This test generates two bitmaps, intersects them, and verifies the result of AVX and SSE.
     */

    const uint64_t allUnset = std::numeric_limits<uint64_t>::min();
    const uint64_t allSet = std::numeric_limits<uint64_t>::max();

    // test column in which all bits are set
    auto testDataColumnSorted =
            generate_exact_number(
                TEST_DATA_COUNT,
                TEST_DATA_COUNT,
                allSet,
                allUnset,
                true
            );
    // test column in which all bits are unset
    auto testDataColumnSorted2 =
            generate_exact_number(
                TEST_DATA_COUNT,
                TEST_DATA_COUNT,
                allUnset,
                allSet,
                true
            );

    auto bitmapAllSet = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(testDataColumnSorted);
    auto bitmapAllUnSet = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(testDataColumnSorted2);

    // result of intersection is ALL BITS UNSET, i.e. TEST_DATA_COUNT uint64_t elements with 0
    auto result =
            intersect_sorted<
                avx2<v256<uint64_t>>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>
            >( bitmapAllSet, bitmapAllUnSet );

    auto result1 =
            intersect_sorted<
                sse<v128<uint64_t>>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>
            >( bitmapAllSet, bitmapAllUnSet );

    const bool allGood =
            memcmp(result->get_data(),result1->get_data(), result1->get_count_values()*8 );

    return allGood;
}