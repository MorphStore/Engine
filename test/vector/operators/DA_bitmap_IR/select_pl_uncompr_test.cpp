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
 * @file select_pl_uncompr_test.cpp
 * @brief Test of vectorized uncompressed select-operator with position-list as intermediate data output.
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>

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

#include <core/operators/general_vectorized/select_pl_uncompr.h>

#include <core/utils/printing.h>

using namespace morphstore;
using namespace vectorlib;

int main( void ) {
    const uint64_t predicate = 250;

    std::cout << "Generating..." << std::flush;
    auto inCol = generate_with_distr(
            1000,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    10000 - 1
            ),
            false
    );
    std::cout << "Done...\n";

    auto result =
            morphstore::select<
                greater,
                avx2<v256<uint64_t>>,
                position_list_f<uncompr_f>,
                uncompr_f
            >(inCol, predicate);

    auto result1 =
            morphstore::select<
                greater,
                sse<v128<uint64_t>>,
                position_list_f<uncompr_f>,
                uncompr_f
            >(inCol, predicate);

    const bool allGood =
            memcmp(result->get_data(),result1->get_data(),result1->get_count_values()*8);

    return allGood;
}