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
#include <core/morphing/default_formats.h>
#include <core/morphing/delta.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/format.h>
#include <core/morphing/morph_batch.h>
#include <core/morphing/morph_saving_offsets.h>
#include <core/morphing/uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/storage/column_with_blockoffsets.h>
#include <core/utils/basic_types.h>
#include <core/utils/equality_check.h>
#include <core/utils/printing.h>

#include <assert.h>

using namespace morphstore;
using namespace vectorlib;

using ve = scalar<v64<uint64_t>>;
using compr_f = DEFAULT_DELTA_DYNAMIC_VBP_F(ve);

int main(void) {
    // 3 whole blocks
    // TODO: also check for partial block (2 variants)
    auto orig_column_size = 3000;
    auto orig_col = generate_sorted_unique(orig_column_size);

    auto compr_col_with_offsets = morph_saving_offsets<ve, compr_f, uncompr_f>(orig_col);

    assert(compr_col_with_offsets->get_block_offsets()->size() ==
           round_up_div(orig_column_size, compr_f::m_BlockSize));
    assert(compr_col_with_offsets->last_block_compressed() == (orig_column_size % compr_f::m_BlockSize == 0));

    // !! currently not equal, as morph_batch on delta block start values very likely depend on previous block
    // delta: morphing multi blocks at once -> block start value = diff to previous block
    //        morphing one block at a time -> block start value = first value of the block
    // example: col 0..2047
    //                --> morph(): start-values: 0 ; 1
    //                --> morph_saving_offsets(): start-values: 0 ; 1024
    /*     std::cout << "Checking morph_saving_offset() result column equals the one from morph()" << std::endl;
        auto compr_col = morph<ve, compr_f, uncompr_f>(orig_col);
        assert_columns_equal(compr_col, compr_col_with_offsets->get_column()); */

    // TODO: get this one to work !! -> block-wise decompression needed
    // currently BUG: 0. block: ok , 1. block: +1023, 2. block: + 3070
    std::cout << "Checking morph_saving_offset() decompressed equals original column" << std::endl;
    auto decompr_col = morph_saving_offsets<ve, uncompr_f, compr_f>(compr_col_with_offsets);
    // uncompr_f blocksize == 1 --> no need to save block offsets
    assert(decompr_col->get_block_offsets()->size() == 0);
    assert_columns_equal(orig_col, decompr_col->get_column());

    return 0;
}