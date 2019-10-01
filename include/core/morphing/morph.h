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
 * @file morph.h
 * @brief Interfaces and useful partial specializations of the morph-operator.
 * 
 * The purpose of the morph-operator is to change the physical representation
 * of some date from a source format to a destination format. This operator
 * exists at two levels: the column-level and the batch-level.
 * 
 * The column-level operator `morph_t` processes entire columns and has to care
 * for the subdivision of a column into a compressed main part and an
 * uncompressed rest part. Note that **there are partial template
 * specializations for compression and decompression** at the column-level, so
 * they do not need to be reimplemented for new formats. These specializations
 * delegate the work to the batch-level morph-operator for the compressable
 * part of the column and take care for the uncompressable rest as well.
 * 
 * The batch-level operator `morph_batch_t` processes plain buffers of data. It
 * can assume that the number of data elements it gets is compressable without
 * an uncompressed rest. This operator is meant to be used whenever parts of a
 * column's data buffer must be morphed separately, e.g., within the
 * column-level morph-operator, the write iterators, or within the batch-level
 * morph-operator of some container format.
 */

#ifndef MORPHSTORE_CORE_MORPHING_MORPH_H
#define MORPHSTORE_CORE_MORPHING_MORPH_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>

#include <cstdint>
#include <cstring>

namespace morphstore {

// ****************************************************************************
// Batch-level
// ****************************************************************************

template<
        class t_vector_extension, class t_dst_f, class t_src_f
>
struct morph_batch_t {
    static void apply(
            // @todo Maybe we should use base_t instead of uint8_t. This could
            // save us some casting in several places.
            const uint8_t * & in8, uint8_t * & out8, size_t countLog
    ) = delete;
};

template<
        class t_vector_extension, class t_dst_f, class t_src_f
>
void morph_batch(const uint8_t * & in8, uint8_t * & out8, size_t countLog) {
    return morph_batch_t<t_vector_extension, t_dst_f, t_src_f>::apply(
            in8, out8, countLog
    );
}


// ****************************************************************************
// Column-level
// ****************************************************************************
    
// ----------------------------------------------------------------------------
// General interface
// ----------------------------------------------------------------------------

/**
 * @brief A struct wrapping the actual morph-operator.
 * 
 * This is necessary to enable partial template specialization, which is
 * required, since some compressed formats have their own template parameters.
 */
template<
        class t_vector_extension,
        class t_dst_f,
        class t_src_f
>
struct morph_t {
    /**
     * @brief Morph-operator. Changes the (compressed) format of the given
     * column from the source format `t_src_f` to the destination format
     * `t_dst_f` without logically changing the data.
     * 
     * This function is deleted by default, to guarantee that using this struct
     * with a format combination it is not specialized for causes a compiler
     * error, not a linker error.
     * 
     * @param inCol The data represented in the source format.
     * @return The same data represented in the destination format.
     */
    static
    const column<t_dst_f> *
    apply(const column<t_src_f> * inCol) = delete;
};

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
template<
        class t_vector_extension,
        class t_dst_f,
        class t_src_f
>
const column<t_dst_f> * morph(const column<t_src_f> * inCol) {
    return morph_t<t_vector_extension, t_dst_f, t_src_f>::apply(inCol);
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
template<class t_vector_extension, class t_f>
struct morph_t<t_vector_extension, t_f, t_f> {
    static
    const column<t_f> *
    apply(const column<t_f> * inCol) {
        return inCol;
    };
};

/**
 * @brief A template specialization of the morph-operator handling the case
 * when the source and the destination format are both uncompressed.
 * 
 * We need to make this case explicit, since otherwise, the choice of the
 * right partial template specialization is ambiguous for the compiler.
 */
template<class t_vector_extension>
struct morph_t<t_vector_extension, uncompr_f, uncompr_f> {
    static
    const column<uncompr_f> *
    apply(const column<uncompr_f> * inCol) {
        return inCol;
    };
};

// ----------------------------------------------------------------------------
// Partial specialization for all compressing morph operators
// ----------------------------------------------------------------------------

template<class t_vector_extension, class t_dst_f>
struct morph_t<t_vector_extension, t_dst_f, uncompr_f> {
    using src_f = uncompr_f;
    
    static
    const column<t_dst_f> *
    apply(const column<src_f> * inCol) {
        const size_t countLog = inCol->get_count_values();
        const size_t outCountLogCompr = round_down_to_multiple(
                countLog, t_dst_f::m_BlockSize
        );
        const size_t outSizeRestByte = uncompr_f::get_size_max_byte(
                countLog - outCountLogCompr
        );

        const uint8_t * in8 = inCol->get_data();

        auto outCol = new column<t_dst_f>(
                get_size_max_byte_any_len<t_dst_f>(countLog)
        );
        uint8_t * out8 = outCol->get_data();
        const uint8_t * const initOut8 = out8;

        morph_batch<t_vector_extension, t_dst_f, src_f>(
                in8, out8, outCountLogCompr
        );
        const size_t sizeComprByte = out8 - initOut8;

        if(outSizeRestByte) {
            out8 = column<t_dst_f>::create_data_uncompr_start(out8);
            memcpy(out8, in8, outSizeRestByte);
        }

        outCol->set_meta_data(
                countLog, out8 - initOut8 + outSizeRestByte, sizeComprByte
        );

        return outCol;
    }
};

// ----------------------------------------------------------------------------
// Partial specialization for all decompressing morph operators
// ----------------------------------------------------------------------------

template<class t_vector_extension, class t_src_f>
struct morph_t<t_vector_extension, uncompr_f, t_src_f> {
    using dst_f = uncompr_f;

    static
    const column<dst_f> *
    apply(const column<t_src_f> * inCol) {
        const uint8_t * in8 = inCol->get_data();
        const size_t inSizeComprByte = inCol->get_size_compr_byte();
        const size_t inSizeUsedByte = inCol->get_size_used_byte();

        const size_t countLog = inCol->get_count_values();
        const uint8_t * const inRest8 = inCol->get_data_uncompr_start();
        const size_t inCountLogRest = (inSizeComprByte < inSizeUsedByte)
                ? convert_size<uint8_t, uint64_t>(
                        inSizeUsedByte - (inRest8 - in8)
                )
                : 0;
        const size_t inCountLogCompr = countLog - inCountLogRest;

        const size_t outSizeByte = dst_f::get_size_max_byte(countLog);
        auto outCol = new column<dst_f>(outSizeByte);
        uint8_t * out8 = outCol->get_data();

        morph_batch<t_vector_extension, dst_f, t_src_f>(
                in8, out8, inCountLogCompr
        );

        memcpy(
                out8, inRest8, uncompr_f::get_size_max_byte(inCountLogRest)
        );

        outCol->set_meta_data(countLog, outSizeByte);

        return outCol;
    }
};

}

#endif //MORPHSTORE_CORE_MORPHING_MORPH_H
