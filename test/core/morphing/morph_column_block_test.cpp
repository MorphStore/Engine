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
 * @brief Tests morphing blocks based on morph_saving_offsets.
 */

#include <core/memory/mm_glob.h>
#include <core/morphing/default_formats.h>
#include <core/morphing/delta.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/format.h>
#include <core/morphing/morph_saving_offsets.h>
#include <core/morphing/decompress_column_block.h>
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
    // TODO: 2 test variants (column_size 3000 and 3072)
    auto orig_column_size = 3000;

    auto orig_col = generate_sorted_unique(orig_column_size);

    // !! morph saving offsets needs to look if last block can be actually morphed (if not complete -> undefined
    // behaviour?)

    auto compr_col_with_offsets = morph_saving_offsets<ve, compr_f, uncompr_f>(orig_col);
    auto block_count = compr_col_with_offsets->get_block_offsets()->size();
    assert(block_count == round_up_div(orig_column_size, compr_f::m_BlockSize));
    assert(compr_col_with_offsets->last_block_compressed() == (orig_column_size % compr_f::m_BlockSize == 0));

    // asserting correctness of decompressing a single block

    auto block_size = compr_col_with_offsets->get_block_size();

    for (uint64_t block = 0; block < compr_col_with_offsets->get_block_offsets()->size(); block++) {
        auto value_count = block_size;

        if (block == block_count -1 && !compr_col_with_offsets->last_block_compressed()) {
            value_count = compr_col_with_offsets->get_column()->get_count_values() % block_size;
        }

        std::cout << "Checking block " << block << " range: " << block * block_size << " .. "
                  << (block * block_size + block_size) - 1 << std::endl;

        auto decompr_col_block = decompress_column_block<ve, compr_f>(compr_col_with_offsets, block);

        auto expected_col = generate_sorted_unique(value_count, block * 1024);
        assert_columns_equal(expected_col, decompr_col_block);
    }

    // checking decompressing multiple sequentiell blocks
    std::cout << "Checking decompressing multiple blocks " << std::endl;
    auto multiple_col_blocks = decompress_column_blocks<ve, compr_f>(compr_col_with_offsets, 0, block_count - 1);
    assert_columns_equal(orig_col, multiple_col_blocks);

    return 0;
}