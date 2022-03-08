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
 * @file simple_query_bm_pl_compr_test.cpp
 * @brief Simple example query to test compressed query processing using bitmaps and position-list
 *        to verify unified query processing.
 */

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/delta.h>
#include <core/operators/general_vectorized/project_pl_compr.h>
#include <core/operators/general_vectorized/project_pl_uncompr.h>
#include <core/operators/general_vectorized/select_bm_compr.h>
#include <core/operators/general_vectorized/select_pl_uncompr.h>
#include <core/morphing/intermediates/transformations.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <functional>
#include <iostream>
#include <random>

using namespace morphstore;
using namespace vectorlib;

// ****************************************************************************
// * Example query
// ****************************************************************************

/**
 * @brief Query = SELECT baseCol2 WHERE baseCol1 < 150
 *        - Selection using bitmap-operator and delta-encoded bm-ouput (rle does not have a decompress_and_process_batch so far)
 *        - Projection using position-list in uncompressed output
 *        - Morphing of intermediates: bitmap<delta> -> position_list<uncompressed> / position_list<delta>
 */

int main( void ) {

    // Some parameters
    const uint64_t predicate = 150;
    const size_t t_BlockSizeLog_delta = 2048;
    using delta =
    delta_f<
            t_BlockSizeLog_delta,
            avx2<v256<uint64_t>>::vector_helper_t::element_count::value,
            uncompr_f
            >;

    // ************************************************************************
    // * Generation of the synthetic base data
    // ************************************************************************

    std::cout << "Base data generation started... ";
    std::cout.flush();

    const size_t countValues = 100 * 1000 * 1000;
    auto baseCol1 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(0, countValues - 1),
            false
    );

    auto baseCol2 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(0, 10),
            false
    );

    std::cout << "done." << std::endl;

    // ************************************************************************
    // * Query execution
    // ************************************************************************

    using ps_avx2 = avx2<v256<uint64_t>>;

    std::cout << "Query execution started... ";
    std::cout.flush();

    // Positions fulfilling "val < 150" -> delta-encoded bitmap as output
    auto bm_compr =
            select_bm_wit_t<
                less,
                ps_avx2,
                bitmap_f<delta>,
                uncompr_f
            >::apply(baseCol1, predicate);

    // Transformation: bitmap<delta> -> position-list<uncompr>
    auto pl_uncompr =
            morph_t<
                ps_avx2,
                position_list_f<uncompr_f>,
                bitmap_f<delta>
            >::apply(bm_compr);

    // Projection using position-lists
    auto query_result =
            project_pl_wit_t<
                ps_avx2,
                uncompr_f,
                uncompr_f,
                position_list_f<uncompr_f>
            >::apply(baseCol2, pl_uncompr);

    std::cout << "done." << std::endl << std::endl;

    // ************************************************************************
    // * Validation using only pl-operator and uncompressed formats
    // ************************************************************************

    // for validation:
    auto validation_pl_uncompr =
            select<
                less,
                ps_avx2,
                position_list_f<uncompr_f>,
                uncompr_f
            >(baseCol1, predicate);

    auto validation_query_result =
            project_pl<
                ps_avx2,
                uncompr_f,
                uncompr_f,
                position_list_f<uncompr_f>
            >(baseCol2, validation_pl_uncompr);

    const bool allGood =
            memcmp(query_result->get_data(),validation_query_result->get_data(),validation_query_result->get_count_values()*8);

    // Print result, if all good:
    if(!allGood) print_columns(print_buffer_base::decimal, query_result, "SELECT baseCol2 WHERE baseCol1 < 150;");

    return allGood;
}