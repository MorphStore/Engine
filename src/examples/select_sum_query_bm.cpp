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
 * @file select_sum_query_bm.cpp
 * @brief A little example query with a selection on one column and a sum on
 * another column using bitmap processing (mainly in selection).
 */

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/operators/scalar/agg_sum_uncompr.h>
#include <core/operators/scalar/project_bm_uncompr.h>
#include <core/operators/scalar/select_bm_uncompr.h>
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

// SELECT SUM(baseCol2) WHERE baseCol1 = 150

int main( void ) {
    // ************************************************************************
    // * Generation of the synthetic base data
    // ************************************************************************

    std::cout << "Base data generation started... ";
    std::cout.flush();

    const size_t countValues = 100 * 1000 * 1000;
    const column<uncompr_f> * const baseCol1 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(100, 199),
            false
    );
    const column<uncompr_f> * const baseCol2 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(0, 10),
            false
    );

    std::cout << "done." << std::endl;

    // ************************************************************************
    // * Query execution
    // ************************************************************************

    std::cout << "Query execution started... ";
    std::cout.flush();

    // Positions fulfilling "baseCol1 = 150"
    auto i1 =
            select<
                std::equal_to,
                scalar<v64<uint64_t>>,
                bitmap_f<uncompr_f>,
                uncompr_f
            >(baseCol1, 150);
    // Data elements of "baseCol2" fulfilling "baseCol1 = 150"
    auto i2 =
            project<
                scalar<v64<uint64_t>>,
                uncompr_f,
                uncompr_f,
                bitmap_f<uncompr_f>
            >(baseCol2, i1);
    // Sum over the data elements of "baseCol2" fulfilling "baseCol1 = 150"
    auto i3 =
            agg_sum<
                scalar<v64<uint64_t>>,
                uncompr_f
            >(i2);

    std::cout << "done." << std::endl << std::endl;

    // ************************************************************************
    // * Result output
    // ************************************************************************

    print_columns(print_buffer_base::decimal, i3, "SUM(baseCol2)");

    return 0;
}