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
 * @file morph_saving_offset.h
 * @brief based on morph.h, just calling morph_batch_t for every block and saving its offset (if blocksize > 1)
 */

#ifndef MORPHSTORE_CORE_MORPHING_MORPH_SAVING_OFFSETS_H
#define MORPHSTORE_CORE_MORPHING_MORPH_SAVING_OFFSETS_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/morphing/morph_batch.h>
#include <core/storage/column.h>
#include <core/storage/column_with_blockoffsets.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>

#include <cstdint>
#include <cstring>

namespace morphstore {

    // ****************************************************************************
    // Column-level
    // ****************************************************************************

    // ----------------------------------------------------------------------------
    // General interface
    // ----------------------------------------------------------------------------

    /**
     * @brief A struct wrapping the actual morph_saving_offsets-operator.
     *
     * This is necessary to enable partial template specialization, which is
     * required, since some compressed formats have their own template parameters.
     */
    template <class t_vector_extension, class t_dst_f, class t_src_f> struct morph_saving_offsets_t {
        /**
         * @brief Morph_with_offsets-operator. Changes the (compressed) format of the given
         * column from the source format `t_src_f` to the destination format
         * `t_dst_f` without logically changing the data.
         *
         * This function is deleted by default, to guarantee that using this struct
         * with a format combination it is not specialized for causes a compiler
         * error, not a linker error.
         *
         * @param inCol The data represented in the source format + previous block_offsets.
         * @return The same data represented in the destination format.
         */
        static column_with_blockoffsets<t_dst_f> *apply(column_with_blockoffsets<t_src_f> *inCol) = delete;
    };

    /**
     * A convenience function wrapping the morph-saving-offset-operator.
     * ! Only works if the block-size = 1 (as otherwise invalid blockoffsets)
     *
     * Changes the (compressed) format of the given column from the source format
     * `t_src_f` to the destination format `t_dst_f` without logically changing the
     * data.
     *
     * @param inCol The data represented in the source format.
     * @return The same data represented in the destination format.
     */
    template <class t_vector_extension, class t_dst_f, class t_src_f>
    column_with_blockoffsets<t_dst_f> *morph_saving_offsets(const column<t_src_f> *inCol) {
        return morph_saving_offsets<t_vector_extension, t_dst_f, t_src_f>(new column_with_blockoffsets(inCol));
    }

    /**
     * A convenience function wrapping the morph-operator.
     *
     * Changes the (compressed) format of the given column from the source format
     * `t_src_f` to the destination format `t_dst_f` without logically changing the
     * data.
     *
     * @param inCol The data represented in the source format.
     * @return The same data represented in the destination format.
     */
    template <class t_vector_extension, class t_dst_f, class t_src_f>
    column_with_blockoffsets<t_dst_f> *morph_saving_offsets(column_with_blockoffsets<t_src_f> *inCol) {
        return morph_saving_offsets_t<t_vector_extension, t_dst_f, t_src_f>::apply(inCol);
    }

    // ----------------------------------------------------------------------------
    // Partial specialization for morphing from a format to itself
    // ----------------------------------------------------------------------------

    /**
     * @brief A template specialization of the morph-operator handling the case
     * when the source and the destination format are the same.
     *
     * It merely returns the given column without doing any work.
     */
    template <class t_vector_extension, class t_f> struct morph_saving_offsets_t<t_vector_extension, t_f, t_f> {
        static column_with_blockoffsets<t_f> *apply(column_with_blockoffsets<t_f> *inCol) { return inCol; };
    };

    /**
     * @brief A template specialization of the morph-operator handling the case
     * when the source and the destination format are both uncompressed.
     *
     * We need to make this case explicit, since otherwise, the choice of the
     * right partial template specialization is ambiguous for the compiler.
     */
    template <class t_vector_extension> struct morph_saving_offsets_t<t_vector_extension, uncompr_f, uncompr_f> {
        static column_with_blockoffsets<uncompr_f> *apply(column_with_blockoffsets<uncompr_f> *inCol) {
            return inCol;
        };
    };

    // ----------------------------------------------------------------------------
    // Partial specialization for all compressing morph operators
    // ----------------------------------------------------------------------------

    template <class t_vector_extension, class t_dst_f>
    struct morph_saving_offsets_t<t_vector_extension, t_dst_f, uncompr_f> {
        using src_f = uncompr_f;

        // saving the offsets for every value would have an unacceptable overhead
        static_assert(t_dst_f::m_BlockSize != 1,
                      "Blocksize of 1 is only expected for uncompr_f .. block-wise morph is useless in this case");

        static column_with_blockoffsets<t_dst_f> *apply(column_with_blockoffsets<src_f> *inCol_with_offsets) {

            const size_t t_BlockSize = t_dst_f::m_BlockSize;

            auto inCol = inCol_with_offsets->get_column();

            std::vector<const uint8_t *> *block_offsets = new std::vector<const uint8_t *>();

            const size_t countLog = inCol->get_count_values();
            const size_t outCountLogCompr = round_down_to_multiple(countLog, t_BlockSize);
            const size_t outSizeRestByte = uncompr_f::get_size_max_byte(countLog - outCountLogCompr);
            
            const uint8_t *in8 = inCol->get_data();

            auto outCol = new column<t_dst_f>(get_size_max_byte_any_len<t_dst_f>(countLog));
            uint8_t *out8 = outCol->get_data();
            const uint8_t *const initOut8 = out8;

            const size_t countBlocks = countLog / t_BlockSize;
            block_offsets->reserve(countBlocks);
            
            // morphing each block and save the offset
            for (size_t blockIdx = 0; blockIdx < countBlocks; blockIdx++) {
                // saving the start address of the block
                block_offsets->push_back(out8);

                // only t_BlockSizeLog as only on block at a time should be morphed
                morph_batch<t_vector_extension, t_dst_f, src_f>(in8, out8, t_BlockSize);
            }

            const size_t sizeComprByte = out8 - initOut8;

            // needed for last block (if incomplete data stays uncompressed)
            if (outSizeRestByte) {
                out8 = column<t_dst_f>::create_data_uncompr_start(out8);
                block_offsets->push_back(out8);
                memcpy(out8, in8, outSizeRestByte);
            }

            outCol->set_meta_data(countLog, out8 - initOut8 + outSizeRestByte, sizeComprByte);

            return new column_with_blockoffsets<t_dst_f>(outCol, block_offsets);
        }
    };

    // ----------------------------------------------------------------------------
    // Partial specialization for all decompressing morph operators
    // ----------------------------------------------------------------------------

    // as uncompressed has a blocksize of 1 --> no need to save blockoffsets
    template <class t_vector_extension, class t_src_f>
    struct morph_saving_offsets_t<t_vector_extension, uncompr_f, t_src_f> {
        using dst_f = uncompr_f;

        static column_with_blockoffsets<dst_f> *apply(column_with_blockoffsets<t_src_f> *inCol_with_offset) {
            // TODO: morph_batch each block independently (see above)
            auto inCol = inCol_with_offset->get_column();
            auto block_offsets = inCol_with_offset->get_block_offsets();

            const size_t countLog = inCol->get_count_values();

            const size_t outSizeByte = dst_f::get_size_max_byte(countLog);
            auto outCol = new column<dst_f>(outSizeByte);
            uint8_t *out8 = outCol->get_data();

            // !! need to decompress each block seperatly
            // example problem:
            // delta: morphing multi blocks at once -> block start value = diff to previous block
            //        morphing one block at a time -> block start value = first actual value of the block
            // example: col 0..2047
            //                --> morph(): start-values: 0 ; 1
            //                --> morph_saving_offsets(): start-values: 0 ; 1024
            for (uint64_t i = 0; i < block_offsets->size(); i++) {
                auto offset = block_offsets->at(i);

                // uncompressed last block
                if ((i == block_offsets->size() - 1) && !inCol_with_offset->last_block_compressed()) {
                    memcpy(out8, offset, uncompr_f::get_size_max_byte(countLog % t_src_f::m_BlockSize));
                } else {
                    morph_batch<t_vector_extension, dst_f, t_src_f>(offset, out8, t_src_f::m_BlockSize);
                }
            }

            outCol->set_meta_data(countLog, outSizeByte);

            return new column_with_blockoffsets(outCol);
        }
    };

} // namespace morphstore

#endif // MORPHSTORE_CORE_MORPHING_MORPH_SAVING_OFFSETS_H
