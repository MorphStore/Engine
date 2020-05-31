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
 * @file morph_saving_offsets_test.cpp
 * @brief Tests morph_saving_offsets.
 */

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/morphing/morph_saving_offsets.h>
#include <core/morphing/default_formats.h>
#include <core/morphing/delta.h>
#include <core/morphing/dynamic_vbp.h>

#include <assert.h>

using namespace morphstore;
using namespace vectorlib;

using ve = scalar<v64<uint64_t>>; 

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    auto origCol = generate_sorted_unique(3000);

    auto col_with_offsets = morph_saving_offsets<ve, DEFAULT_DELTA_DYNAMIC_VBP_F(ve), uncompr_f>(origCol);

    assert(col_with_offsets->block_offsets->size() == 3);

    return false;
}