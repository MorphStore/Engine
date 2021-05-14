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
 * @file column_gen_test.cpp
 * @brief A small test and example usage of some functions for initializing
 * uncompressed columns (from column_gen.h).
 * @todo TODOS?
 */

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>

#include <cstdint>
#include <random>

using namespace morphstore;

int main( void ) {
    const size_t countValues = 20;
    
    auto col0 = ColumnGenerator::make_column({2, 3, 5, 7, 11, 13});
    auto col1 = generate_sorted_unique(countValues);
    auto col2 = generate_sorted_unique(countValues, 100 * 1000, 1000);
    auto col3 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(100, 200),
            false
    );
    auto col4 = generate_with_distr(
            countValues,
            std::discrete_distribution<uint64_t>({0, 0, 1, 0, 7, 2}),
            true
    );
    
    print_columns(
            print_buffer_base::decimal,
            col0,
            col1,
            col2,
            col3,
            col4,
            "some hardcoded values",
            "sorted ids",
            "other sorted ids",
            "some uniform distribution (unsorted)",
            "some discrete distribution (sorted)"
    );
    
    return 0;
}
