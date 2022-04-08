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
 * @file select_bm_compr_test.cpp
 * @brief Test of vectorized compressed select-operator with bitmap as intermediate data output.
 *        => Generally to test decompress_and_process_batch with bitmap_f<> as format
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/delta.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/printing.h>

#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>



#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include <vector/simd/sse/primitives/compare_sse.h>

#include <core/operators/general_vectorized/select_bm_compr.h>
#include <core/operators/general_vectorized/select_bm_uncompr.h>

#define TEST_DATA_COUNT 1000

using namespace morphstore;
using namespace vectorlib;

int main( void ) {

    /** General idea:
     *                 (1)   Generate uncompressed data -> inCol
     *                 (1.1) Process inCol with uncompressed-select-operator (select_bm_uncompr.h) to compare with
     *                       compressed select operator output later in step 4
     *                 (2)   Compress inCol to delta_f to have compressed input for select_bm_compr-operator
     *                 (3)   Process delta_f-inCol with compressed-select-operator
     *                 (4)   Compare results from step 1.1 and step 3
     */

    // Some parameters
    const uint64_t predicate = 250;
    const size_t t_BlockSizeLog_delta = 2048;
    using delta = delta_f<
            t_BlockSizeLog_delta,
            avx2<v256<uint64_t>>::vector_helper_t::element_count::value,
            uncompr_f>;

    // (1) Generate uncompressed data
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

    // (1.1) uncompr-select-operator: input -> uncompr_f, output -> bitmap_f<uncompr_f>
    auto result_uncompr =
            morphstore::select<
                greater,
                sse<v128<uint64_t>>,
                bitmap_f<uncompr_f>,
                uncompr_f
            >(inCol, predicate);

    // (2) Compress data to delta_f
    auto inCol_delta =
            morph_t<
                avx2<v256<uint64_t>>,
                delta,
                uncompr_f
            >::apply(inCol);

    // (3) Select operator: input -> delta , output -> bitmap_f<uncompr_f>
    auto result_compr =
            select_bm_wit_t<
                greater,
                avx2<v256<uint64_t>>,
                bitmap_f<uncompr_f>,
                delta
            >::apply(inCol_delta, predicate);

    //print_columns(print_buffer_base::decimal, result_compr, "result_compr");

    // (4) Compare results
    const bool allGood =
            memcmp(result_compr->get_data(),result_uncompr->get_data(),result_uncompr->get_count_values()*8);

    return allGood;
}