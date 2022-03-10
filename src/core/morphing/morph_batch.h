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
 * @file morph_batch.h
 * @brief Interfaces and useful partial specializations of the batch-level
 * morph-operator.
 * 
 * See the documentation of morph.h for general information on the
 * morph-operator.
 */

#ifndef MORPHSTORE_CORE_MORPHING_MORPH_BATCH_H
#define MORPHSTORE_CORE_MORPHING_MORPH_BATCH_H

#include <core/morphing/format.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>

#include <cstdint>
#include <cstring>

namespace morphstore {

// ****************************************************************************
// Batch-level
// ****************************************************************************
    
// ----------------------------------------------------------------------------
// General interface
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Partial specialization for morphing from uncompressed to uncompressed
// ----------------------------------------------------------------------------

template<class t_vector_extension>
struct morph_batch_t<t_vector_extension, uncompr_f, uncompr_f> {
    static void apply(
            const uint8_t * & in8, uint8_t * & out8, size_t countLog
    ) {
        const size_t sizeByte = convert_size<uint64_t, uint8_t>(countLog);
        memcpy(out8, in8, sizeByte);
        in8 += sizeByte;
        out8 += sizeByte;
    };
};

}

#endif //MORPHSTORE_CORE_MORPHING_MORPH_BATCH_H
