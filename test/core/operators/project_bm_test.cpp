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
 * @file project_bm_test.cpp
 * @brief A little test/reference of the project-operator using bitmaps.
 * @todo TODOS?
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
#include <core/operators/scalar/project_bm_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <vector/scalar/extension_scalar.h>
#include <core/utils/printing.h>
#include <core/utils/equality_check.h>

using namespace morphstore;
using namespace vectorlib;

int main( void ) {
    auto inCol = make_column({11, 44, 22, 33, 11});

    // bitmap column with encoded positions at 1,3 and 4 => 11010 -> (int) 26
    auto bitmapCol = reinterpret_cast<const column< bitmap_f<uncompr_f> > *>(
            make_column({26})
    );

    // project operation
    auto outCol =
            project<
                scalar<v64<uint64_t>>,
                uncompr_f,
                uncompr_f,
                bitmap_f<uncompr_f>
            >(inCol, bitmapCol);

    auto OutColExp = make_column({44, 33, 11});

    const equality_check ec(OutColExp, outCol);
    const bool allGood = ec.good();

    return !allGood;
}