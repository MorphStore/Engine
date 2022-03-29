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
 * @file write_iterator_IR_pl_test.cpp
 * @brief Testing write_iterator_IR implementation that introduces intermediate
 *        representation transformations (IR-Transformations) in the output-side-buffer-layer.
 *
 *        Test specifically for PL-transformations, i.e. where position_list_f is src-format.
 *
 *        For more details, see /core/morphing/write_iterator_IR.h.
 *
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/printing.h>
#include <core/utils/basic_types.h>

#include <core/morphing/write_iterator_IR.h>
#include <core/morphing/intermediates/transformations/transformation_algorithms.h>

#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>

#include <core/operators/general_vectorized/select_pl_compr.h>
#include <core/operators/general_vectorized/select_bm_uncompr.h>

#include <core/morphing/uncompr.h>

#include <iostream>
#include <type_traits>

#define TEST_DATA_COUNT 1000

using namespace morphstore;
using namespace vectorlib;

int main(void) {

    /** General idea: Always execute 2 select-queries with uncompressed output
     *                (1) using vectorized-PL-compressed-operator
     *                (2) using vectorized-BM-uncompressed-operator
     *
     *                => Finally, compare results to see if transformation is done correctly within write_iterator
     *
     *                This process is repeated for the following internal IR-transformation cases:
     *                  (i)   position-list -> bitmaps
     *                  (ii)  position-list -> position-list
     *
     */

    const uint64_t predicate = 250;

    // (1) Generate uncompressed base data
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

    // **************************** (i) position-list -> bitmaps ****************************
    auto result_compr_1 =
            select_pl_wit_t<
                greater,
                avx2<v256<uint64_t>>,
                bitmap_f<uncompr_f>,
                uncompr_f
            >::apply(inCol, predicate);

    auto result_uncompr_1=
            morphstore::select<
                greater,
                avx2<v256<uint64_t>>,
                bitmap_f<uncompr_f>,
                uncompr_f
            >(inCol, predicate);

    //print_columns(print_buffer_base::binary, result_compr_1, "result_compr");
    //print_columns(print_buffer_base::binary, result_uncompr_1, "result_uncompr");

    const bool allGood_1 =
            memcmp(result_compr_1->get_data(), result_uncompr_1->get_data(), result_uncompr_1->get_count_values()*8);

    // **************************** (ii) position-list -> position-list ****************************
    auto result_compr_2 =
            select_pl_wit_t<
                greater,
                avx2<v256<uint64_t>>,
                position_list_f<uncompr_f>,
                uncompr_f
            >::apply(inCol, predicate);

    // to avoid redefinition errors when including select_pl_uncompr.h, we simply transform result_uncompr_1 to a position-list for comparison
    // In general, it is bad to add another component (IR_transformation_algorithms) as this increases dependencies... but for now simplest way
    auto result_uncompr_2 =
            transform_IR<
                avx2<v256<uint64_t>>,
                position_list_f<>,
                bitmap_f<>
            >(result_uncompr_1);

    const bool allGood_2 =
            memcmp(result_compr_2->get_data(), result_uncompr_2->get_data(), result_uncompr_2->get_count_values()*8);

    return allGood_1 || allGood_2;
}