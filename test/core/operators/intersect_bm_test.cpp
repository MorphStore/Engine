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
 * @file intersect_bm_test.cpp
 * @brief A little test/reference of the intersect-operator using bitmaps.
 * @todo TODOS?
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/operators/scalar/intersect_bm_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <vector/scalar/extension_scalar.h>
#include <core/utils/printing.h>

#include <limits>

using namespace morphstore;
using namespace vectorlib;

int main(void) {

    /**
     *  @brief: Testing intersect-operator for bitmaps to enable execution of multidimensional boolean queries.
     *          This test generates two bitmaps, intersects them, and verifies the result.
     */

    const uint64_t unset = std::numeric_limits<uint64_t>::min();
    const uint64_t set = std::numeric_limits<uint64_t>::max();

    // ********************** (1) intersection test **********************
    // inColL -> all bits set; inColR -> all bits unset => intersection-result = all bits unset
    auto inColL_1 = make_column({  set,   set,   set,   set,   set});
    auto inColR_1 = make_column({unset, unset, unset, unset, unset});

    auto inBmLCol_1 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColL_1);
    auto inBmRCol_1 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColR_1);

    auto intersection_1 =
            intersect_sorted<
                vectorlib::scalar<vectorlib::v64<uint64_t>>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>
            >(inBmLCol_1, inBmRCol_1);

    // check if all bits are unset & size equal to min. inCol
    bool allGood_1 = false;
    uint64_t * ptr_1= intersection_1->get_data();
    size_t count_1 = inColL_1->get_count_values();
    while(count_1--){
        if(*ptr_1) allGood_1 = true;
        ptr_1++;
    }

    // ********************** (2) intersection test **********************
    // inColL and inColR consists of alternating words in which all bits are set or unset => intersection-result = all bits unset
    auto inColL_2 = make_column({  set, unset,   set, unset,   set});
    auto inColR_2 = make_column({unset,   set, unset,   set, unset});

    auto inBmLCol_2 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColL_2);
    auto inBmRCol_2 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColR_2);

    auto intersection_2 =
            intersect_sorted<
                vectorlib::scalar<vectorlib::v64<uint64_t>>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>
            >(inBmLCol_2, inBmRCol_2);

    // check if all bits are unset & size equal to min. inCol
    bool allGood_2 = false;
    uint64_t * ptr_2 = intersection_2->get_data();
    size_t count_2 = inColL_2->get_count_values();
    while(count_2--){
        if(*ptr_2) allGood_2 = true;
        ptr_2++;
    }

    // ********************** (3) intersection test **********************
    // in inColL and inColR are all bits set => intersection-result = all bits set
    auto inColL_3 = make_column({set, set, set, set, set});
    auto inColR_3 = make_column({set, set, set, set, set});

    auto inBmLCol_3 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColL_3);
    auto inBmRCol_3 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColR_3);

    auto intersection_3 =
            intersect_sorted<
                vectorlib::scalar<vectorlib::v64<uint64_t>>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>
            >(inBmLCol_3, inBmRCol_3);

    // check if all bits are set & size equal to min. inCol
    bool allGood_3 = false;
    uint64_t * ptr_3 = intersection_3->get_data();
    size_t count_3 = inColL_3->get_count_values();
    while(count_3--){
        if(!*ptr_2) allGood_3 = true;
        ptr_3++;
    }

    return allGood_1 & allGood_2 && allGood_3;
}