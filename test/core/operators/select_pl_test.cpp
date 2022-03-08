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
 * @file select_pl_test.cpp
 * @brief Small test of the select-operator using position-lists.
 * @todo TODOS?
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
#include <core/operators/scalar/select_pl_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <vector/scalar/extension_scalar.h>
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>
#include <core/utils/equality_check.h>

using namespace morphstore;
using namespace vectorlib;

int main( void ) {
    auto inCol = make_column({95, 102, 100, 87, 120});

    // select operation - bitmask_f<uncompr_f> output
    auto outCol =
            select<
                std::less,
                scalar<v64<uint64_t>>,
                position_list_f<uncompr_f>,
                uncompr_f
            >(inCol, 100);

    // inCol-values < 100: positions={0, 3}
    auto OutColExp = reinterpret_cast< const column< position_list_f<uncompr_f> > * >(
            make_column({0,3})
    );

    const equality_check ec(OutColExp, outCol);
    const bool allGood = ec.good();

    return !allGood;
}