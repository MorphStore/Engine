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
 * @file merge_bm_test.cpp
 * @brief A little test/reference of the merge-operator using bitmaps.
 * @todo TODOS?
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/operators/scalar/merge_bm_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <vector/scalar/extension_scalar.h>
#include <core/utils/printing.h>

#include <limits>

using namespace morphstore;
using namespace vectorlib;

int main(void) {

    /**
     *  @brief: Testing merge-operator for bitmaps to enable execution of multidimensional boolean queries.
     *          This test generates two bitmaps, merges them, and verifies the result.
     */

    const uint64_t allUnset = std::numeric_limits<uint64_t>::min();
    const uint64_t allSet = std::numeric_limits<uint64_t>::max();

    // ********************** (1) merge test **********************
    // inColL -> all bits set; inColR -> all bits unset => merge-result = all bits set
    auto inColL_1 = make_column({  allSet,   allSet,   allSet,   allSet,   allSet});
    auto inColR_1 = make_column({allUnset, allUnset, allUnset, allUnset, allUnset});

    auto inBmLCol_1 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColL_1);
    auto inBmRCol_1 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColR_1);

    auto merge_1 =
            merge_sorted<
                vectorlib::scalar<vectorlib::v64<uint64_t>>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>
            >(inBmLCol_1, inBmRCol_1);

    // check if all bits are set
    bool allGood_1 = false;
    uint64_t * ptr_1 = merge_1->get_data();
    size_t count_1 = inColL_1->get_count_values();
    while(count_1--){
        if(!*ptr_1) allGood_1 = true;
        ptr_1++;
    }

    // ********************** (2) merge test **********************
    // inColL and inColR consists of alternating words in which all bits are set or unset => merge-result = all bits set
    auto inColL_2 = make_column({  allSet, allUnset,   allSet, allUnset,   allSet});
    auto inColR_2 = make_column({allUnset,   allSet, allUnset,   allSet, allUnset});

    auto inBmLCol_2 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColL_2);
    auto inBmRCol_2 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColR_2);

    auto merge_2 =
            merge_sorted<
                vectorlib::scalar<vectorlib::v64<uint64_t>>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>
            >(inBmLCol_2, inBmRCol_2);

    // check if all bits are unset & size equal to min. inCol
    bool allGood_2 = false;
    uint64_t * ptr_2 = merge_2->get_data();
    size_t count_2 = inColL_2->get_count_values();
    while(count_2--){
        if(!*ptr_2) allGood_2 = true;
        ptr_2++;
    }

    // ********************** (3) merge test **********************
    // in inColL and inColR are all bits unset => merge-result = all bits unset
    auto inColL_3 = make_column({allUnset, allUnset, allUnset, allUnset, allUnset});
    auto inColR_3 = make_column({allUnset, allUnset, allUnset, allUnset, allUnset});

    auto inBmLCol_3 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColL_3);
    auto inBmRCol_3 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColR_3);

    auto merge_3 =
            merge_sorted<
                vectorlib::scalar<vectorlib::v64<uint64_t>>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>
            >(inBmLCol_3, inBmRCol_3);

    // check if all bits are unset & size equal to min. inCol
    bool allGood_3 = false;
    uint64_t * ptr_3 = merge_3->get_data();
    size_t count_3 = inColL_3->get_count_values();
    while(count_3--){
        if(*ptr_3) allGood_3 = true;
        ptr_3++;
    }

    // ********************** (4) merge test **********************
    // two input bimtaps of different lengths -> output has to be the same as the larger one
    // + the resulting bits are exactly the same as the larger ones, i.e. between end of small and to the end of largest
    auto inColL_4 = make_column({allSet, allSet, allSet, allSet, allSet, allUnset, allUnset, allUnset});
    auto inColR_4 = make_column({allSet, allSet, allSet, allSet, allSet});

    auto inBmLCol_4 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColL_4);
    auto inBmRCol_4 = reinterpret_cast< const column< bitmap_f<uncompr_f> > * >(inColR_4);

    auto merge_4 =
            merge_sorted<
                vectorlib::scalar<vectorlib::v64<uint64_t>>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>,
                bitmap_f<uncompr_f>
            >(inBmLCol_4, inBmRCol_4);

    // check if all bits are set to the end of smaller size bitmap (right one)
    bool allGood_4 = false;
    uint64_t * ptr_4 = merge_4->get_data();
    size_t count_4 = inColR_4->get_count_values();
    while(count_4--){
        if(!*ptr_4) allGood_4 = true;
        ptr_4++;
    }
    // check remaining bits of output -> they all have to be unset
    size_t count_delta_4 = inColR_4->get_count_values() - inColR_4->get_count_values();
    while(count_delta_4--){
        if(*ptr_4) allGood_4 = true;
        ptr_4++;
    }

    return allGood_1 & allGood_2 && allGood_3 && allGood_4;
}