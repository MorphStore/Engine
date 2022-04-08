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
 * @file project_pl_compr_test.h
 * @brief Test for vectorized project operator using compressed position-lists and compressed input.
 *        => Generally to test decompress_and_process_batch with position_list_f<> as format
 *
 */

#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
#include <core/morphing/delta.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/intermediates/position_list.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/printing.h>

#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>



#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>

#include <core/operators/general_vectorized/project_pl_compr.h>
#include <core/operators/general_vectorized/project_pl_uncompr.h>

#include <iostream>

#define TEST_DATA_COUNT 1000

using namespace morphstore;
using namespace vectorlib;

int main( void ) {

    /** General idea:
     *                 (1)   Generate uncompressed data -> inCol + uncompressed position-list inPosCol
     *                 (1.1) Process inCol with uncompressed-project-operator (project_pl_uncompr.h) to compare
     *                       with compressed project operator output later in step 4
     *                 (2)   Compress inPosCol to delta_f to have compressed IR-input for project_pl_compr-operator
     *                 (3)   Process delta-inPosCol with compressed-project-operator
     *                 (4)   Compare results from step 1.1 and step 3
     */

    const size_t t_BlockSizeLog_delta = 2048;
    using delta = delta_f<
            t_BlockSizeLog_delta,
            avx2<v256<uint64_t>>::vector_helper_t::element_count::value,
            uncompr_f>;

    // (1) Generate uncompressed data + uncompressed position-list
    std::cout << "Generating..." << std::flush;
    auto inCol_data = generate_with_distr(
            TEST_DATA_COUNT,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    TEST_DATA_COUNT - 1
            ),
            false
    );
    // position-list (uncompressed)
    auto inCol_position_list_uncompr = reinterpret_cast<const column< position_list_f<uncompr_f> > *>(
            generate_sorted_unique(TEST_DATA_COUNT,0,1)
    );
    std::cout << "Done...\n";

    // (1.1) uncompr-project-operator: input -> uncompr_f, output -> position_list_f<uncompr_f>
    auto result_uncompr =
            morphstore::project_pl<
                sse<v128<uint64_t>>,
                uncompr_f,
                uncompr_f,
                position_list_f<uncompr_f>
            >(inCol_data, inCol_position_list_uncompr);

    // (2) Compress position-list to delta
    auto inCol_position_list_delta =
            morph_t<
                avx2<v256<uint64_t>>,
                position_list_f<delta>,
                position_list_f<uncompr_f>
            >::apply(inCol_position_list_uncompr);

    // (3) Project operator: input -> position_list<delta> , uncompr_f
    auto result_compr =
            project_pl_wit_t<
                avx2<v256<uint64_t>>,
                uncompr_f,
                uncompr_f,
                position_list_f<>, // dest IR-type
                position_list_f<delta> // src IR-type
            >::apply(inCol_data, inCol_position_list_delta);

    //print_columns(print_buffer_base::decimal, result_compr, "result_compr");

    // (4) Compare results
    const bool allGood =
            memcmp(result_uncompr->get_data(),result_compr->get_data(),result_uncompr->get_count_values()*8);

    return allGood;
}