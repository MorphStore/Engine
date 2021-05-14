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
 * @file agg_sum_group_based_test.cpp
 * @brief A little test/reference of the group-based aggregation(sum)-operator.
 * @todo TODOS?
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include "operator_test_frames.h"
#include <core/morphing/format.h>
#include <core/operators/scalar/agg_sum_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <vector/scalar/extension_scalar.h>

using namespace morphstore;
using namespace vectorlib;

int main( void ) {
    const bool allGood = test_op_2in_1out_1val(
            "Group-Based aggregation(sum)",
            &agg_sum<scalar<v64<uint64_t>>, uncompr_f>,
            ColumnGenerator::make_column({0, 0, 1, 0, 2, 1}),
            ColumnGenerator::make_column({100, 150, 50, 500, 200, 100}),
            "inGrCol",
            "inDataCol",
            ColumnGenerator::make_column({750, 150, 200}),
            "outDataCol",
            3
    );
    
    return !allGood;
}
