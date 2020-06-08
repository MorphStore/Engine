/**********************************************************************************************
 * Copyright (C) 2020 by MorphStore-Team                                                      *
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
 * @file decompress_column_block.h
 * @brief Decompressing column blocks based on column_with_blockoffsets.
 */

#ifndef MORPHSTORE_CORE_MORPHING_DECOMPRESS_COLUMN_BLOCK_H
#define MORPHSTORE_CORE_MORPHING_DECOMPRESS_COLUMN_BLOCK_H

#include <core/morphing/morph_batch.h>
#include <core/storage/column.h>
#include <core/storage/column_with_blockoffsets.h>

#include <assert.h>

namespace morphstore {

    /**
     * @brief Decompressing a range column blocks (inclusive range)
     *
     * @param inCol The column with block-offsets
     * @param start index of blocks to be decompressed
     * @param end index of blocks to be decompressed
     * @return Specified blocks uncompressed in a new column
     */
    template <class ve, class compr_f>
    const column<uncompr_f> *decompress_column_blocks(column_with_blockoffsets<compr_f> *inCol, uint64_t start,
                                                      uint64_t end) {
        static_assert(compr_f::m_BlockSize != 1, "Decompressing column blocks of size 1 is not allowed");

        auto block_size = compr_f::m_BlockSize;
        auto block_count = inCol->get_block_offsets()->size();
        auto inCol_value_count = inCol->get_column()->get_count_values();

        // validating range
        assert(start <= end);
        assert(start <= block_count);
        assert(end <= block_count);

        bool last_block_uncompressed = !inCol->last_block_compressed();
        bool last_block_included = end == (block_count - 1);

        // pessimistic value_count (assuming all blocks are complete)
        auto value_count = (end - start + 1) * block_size;

        if (last_block_included && last_block_uncompressed) {
            // correcting value_count estimation
            value_count -= block_size - inCol_value_count % block_size;
        }

        // TODO: should actually be base_t?
        auto alloc_size = value_count * sizeof(uint64_t);

        auto decompr_col_blocks = new column<uncompr_f>(alloc_size);
        decompr_col_blocks->set_meta_data(value_count, alloc_size);
        uint8_t *out8 = decompr_col_blocks->get_data();

        for (uint64_t block = start; block <= end; block++) {
            const uint8_t *block_offset = inCol->get_block_offset(block);

            if (block == end && last_block_included && last_block_uncompressed) {
                // handle uncompressed part
                morph_batch<ve, uncompr_f, uncompr_f>(block_offset, out8, inCol_value_count % block_size);
            } else {
                morph_batch<ve, uncompr_f, compr_f>(block_offset, out8, block_size);
            }
        }

        return decompr_col_blocks;
    }

    template <class ve, class compr_f>
    const column<uncompr_f> *decompress_column_block(column_with_blockoffsets<compr_f> *inCol, uint64_t block_index) {
        return decompress_column_blocks<ve, compr_f>(inCol, block_index, block_index);
    }

} // namespace morphstore

#endif // MORPHSTORE_CORE_MORPHING_DECOMPRESS_COLUMN_BLOCK_H
