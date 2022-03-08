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
 * @file IR_transformation_uncompr_test.cpp
 * @brief Small test of the morph operator for uncompressed intermediate data, i.e. position-list -> bitmap and
 *        bitmap -> position-list, using vectorized (avx2) processing.
 *
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/printing.h>
#include <core/morphing/format.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/morphing/intermediates/transformations.h>

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
    // * Morph-Operations - uncompressed IR-Transformation
    // ************************************************************************

    /** General idea:
     *                 (1) morph: position_list_1 -> bitmap_1
     *                 (2) morph: bitmap_1 -> position_list_2
     *                 (3) compare: position_list_1 == position_list_2 ?
     */

    auto pl_uncompr_1 = reinterpret_cast< const column< position_list_f<uncompr_f> > * >(
            generate_sorted_unique(TEST_DATA_COUNT,0,1)
    );

    // ***** (1) morph: position_list_1 -> bitmap_1 *****
    auto bm_uncompr_1 =
            morph_t<
                avx2<v256<uint64_t>>,
                bitmap_f<>,
                position_list_f<>
            >::apply(pl_uncompr_1);

    // ***** (2) morph: bitmap_1 -> position_list_2 *****
    auto pl_uncompr_2 =
            morph_t<
                avx2<v256<uint64_t>>,
                position_list_f<>,
                bitmap_f<>
            >::apply(bm_uncompr_1);

    // ***** (3) compare: position_list_1 == position_list_2 ? *****
    const bool allGood =
            memcmp(pl_uncompr_1->get_data(), pl_uncompr_2->get_data(), (int)(TEST_DATA_COUNT*8)); //returns zero if all bytes match

    return allGood;
}