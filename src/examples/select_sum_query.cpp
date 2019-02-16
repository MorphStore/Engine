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
 * @file select_sum_query.cpp
 * @brief A little example query with a selection on one column and a sum on
 * another column.
 * @todo TODOS?
 */

#include "../../include/core/memory/mm_glob.h"
#include "../../include/core/morphing/format.h"
#include "../../include/core/operators/scalar/agg_sum_uncompr.h"
#include "../../include/core/operators/scalar/project_uncompr.h"
#include "../../include/core/operators/scalar/select_uncompr.h"
#include "../../include/core/storage/column.h"
#include "../../include/core/storage/column_gen.h"
#include "../../include/core/utils/basic_types.h"
#include "../../include/core/utils/printing.h"
#include "../../include/core/utils/processing_style.h"

#include <functional>
#include <iostream>
#include <random>

using namespace morphstore;

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
    
    const processing_style_t scalar = processing_style_t::scalar;
    
    std::cout << "Query execution started... ";
    std::cout.flush();
    
    // Positions fulfilling "baseCol1 = 150"
    auto i1 = morphstore::select<
            std::equal_to,
            scalar,
            uncompr_f,
            uncompr_f
    >::apply(baseCol1, 150);
    // Data elements of "baseCol2" fulfilling "baseCol1 = 150"
    auto i2 = project<scalar, uncompr_f>(baseCol2, i1);
    // Sum over the data elements of "baseCol2" fulfilling "baseCol1 = 150"
    auto i3 = agg_sum<scalar, uncompr_f>(i2);
    
    std::cout << "done." << std::endl << std::endl;
    
    // ************************************************************************
    // * Result output
    // ************************************************************************

    print_columns(print_buffer_base::decimal, i3, "SUM(baseCol2)");
    
    return 0;
}