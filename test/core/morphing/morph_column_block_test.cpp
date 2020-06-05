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
    // TODO: also check for partial block .. 2 test variants (column_size 3000 and 3072)
    // 3000 not working yet
    auto column_size = 3000;

    auto orig_col = generate_sorted_unique(column_size);

    // !! morph saving offsets needs to look if last block can be actually morphed (if not complete -> undefined
    // behaviour?)

    auto compr_col_with_offsets = morph_saving_offsets<ve, compr_f, uncompr_f>(orig_col);
    assert(compr_col_with_offsets->get_block_offsets()->size() == 3);
    assert(compr_col_with_offsets->last_block_compressed() == (column_size % compr_f::m_BlockSize == 0));

    // asserting correctness of decompressing a single block
    auto block_size = compr_col_with_offsets->get_block_size();

    for (uint64_t block = 0; block < compr_col_with_offsets->get_block_offsets()->size(); block++) {
        auto value_count = block_size;

        bool last_block = (block == (compr_col_with_offsets->get_block_offsets()->size() - 1));
        bool last_block_uncompressed = !compr_col_with_offsets->last_block_compressed();
        const uint8_t *block_offset = compr_col_with_offsets->get_block_offset(block);

        if (last_block && last_block_uncompressed) {
            value_count = compr_col_with_offsets->get_column()->get_count_values() % block_size;
        }

        std::cout << "Checking block " << block << " range: " << block * block_size << " .. "
                  << (block * block_size + value_count) - 1 << std::endl;

        auto alloc_size = value_count * sizeof(uint64_t);

        // TODO: refactor into general function:
        // checking if block size == 1 (then direct mem_copy)
        // checking if last block -> direct mem_copy + right meta data setting (value count < block_size)
        // column<trg> morph_block<ve, trg, src>(column_with_offset<src> col_with_offsets)

        auto decompr_col_block = new column<uncompr_f>(alloc_size);
        decompr_col_block->set_meta_data(value_count, alloc_size);
        uint8_t *out8 = decompr_col_block->get_data();
    

        if (last_block && last_block_uncompressed) {
            auto outSizeRestByte = uncompr_f::get_size_max_byte(value_count);
            memcpy(out8, block_offset, outSizeRestByte);
        } else {
            morph_batch<ve, uncompr_f, compr_f>(block_offset, out8, block_size);
        }

        auto expected_col = generate_sorted_unique(value_count, block * 1024);
        assert_columns_equal(expected_col, decompr_col_block);
    }

    return 0;
}