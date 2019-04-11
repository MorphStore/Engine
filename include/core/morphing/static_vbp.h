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
 * @file static_vbp.h
 * @brief A compressed format using the vertical bit-packed layout with a fixed
 *        bit width for an entire column and the corresponding morph operators
 *        for compression and decompression.
 */

#ifndef MORPHSTORE_CORE_MORPHING_STATIC_VBP_H
#define MORPHSTORE_CORE_MORPHING_STATIC_VBP_H

#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/morphing/vbp_routines.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/processing_style.h>

#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace morphstore {
    
    // The vertical bit packed format with a static bit width.
    template<unsigned bw>
    struct static_vbp_f : public format {
        static_assert(
           (1 <= bw) && (bw <= std::numeric_limits<uint64_t>::digits),
           "static_vbp: template parameter bw must satisfy 1 <= bw <= 64"
        );
    };
    
    template<unsigned bw>
    struct morph_t<
            processing_style_t::vec128,
            static_vbp_f<bw>,
            uncompr_f
    > {
        using out_f = static_vbp_f<bw>;
        using in_f = uncompr_f;
        
        static
        const column<out_f> *
        apply(const column<in_f> * inCol) {
            // TODO support arbitrary numbers of data elements
            if(inCol->get_count_values() % 128)
                throw std::runtime_error(
                        "morph uncompr_f -> static_vbp_f: the number of data "
                        "elements must be a multiple of 128"
                );
            const __m128i * in128 = inCol->get_data();
            
            auto outCol = new column<out_f>(inCol->get_size_used_byte());
            __m128i * out128 = outCol->get_data();
            const __m128i * const initOut128 = out128;

            pack<bw>(
                    in128,
                    convert_size<uint8_t, __m128i>(
                            inCol->get_size_used_byte()
                    ),
                    out128
            );

            outCol->set_meta_data(
                    inCol->get_count_values(),
                    convert_size<__m128i, uint8_t>(out128 - initOut128)
            );
            
            return outCol;
        }
    };
    
    template<unsigned bw>
    struct morph_t<
            processing_style_t::vec128,
            uncompr_f,
            static_vbp_f<bw>
    > {
        using out_f = uncompr_f;
        using in_f = static_vbp_f<bw>;
        
        static
        const column<out_f> *
        apply(const column<in_f> * inCol) {
            // TODO support arbitrary numbers of data elements
            if(inCol->get_count_values() % 128)
                throw std::runtime_error(
                        "morph uncompr_f -> static_vbp_f: the number of data "
                        "elements must be a multiple of 128"
                );
            const __m128i * in128 = inCol->get_data();
            
            auto outCol = new column<out_f>(
                    convert_size<uint64_t, uint8_t>(inCol->get_count_values())
            );
            __m128i * out128 = outCol->get_data();
            const __m128i * const initOut128 = out128;

            unpack<bw>(
                    in128,
                    out128,
                    convert_size<uint64_t, __m128i>(inCol->get_count_values())
            );

            outCol->set_meta_data(
                    inCol->get_count_values(),
                    convert_size<__m128i, uint8_t>(out128 - initOut128)
            );

            return outCol;
        }
    };
    
}
#endif //MORPHSTORE_CORE_MORPHING_STATIC_VBP_H
