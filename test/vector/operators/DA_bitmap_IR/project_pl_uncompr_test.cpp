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
 * @file project_pl_test.h
 * @brief Test for vectorized project operator using position-lists
 * @todo TODOS?
 */

#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
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

#include <core/operators/general_vectorized/project_pl_uncompr.h>

#include <iostream>

#define TEST_DATA_COUNT 100

using namespace morphstore;
using namespace vectorlib;

int main( void ) {
    std::cout << "Generating..." << std::flush;
    auto inCol_data = generate_sorted_unique(TEST_DATA_COUNT,0,1);
    auto inCol_position_list = reinterpret_cast<const column< position_list_f<uncompr_f> > *>(
            generate_sorted_unique(TEST_DATA_COUNT,0,1)
    );
    std::cout << "Done...\n";

    auto result =
            project_pl<
                avx2<v256<uint64_t>>,
                uncompr_f,
                uncompr_f,
                position_list_f<uncompr_f>
            >(inCol_data, inCol_position_list);

    auto result1 =
            project_pl<
                sse<v128<uint64_t>>,
                uncompr_f,
                uncompr_f,
                position_list_f<uncompr_f>
            >(inCol_data, inCol_position_list);

    const bool allGood =
            memcmp(result->get_data(), result1->get_data(), (int)(TEST_DATA_COUNT*8));

    return allGood;
}