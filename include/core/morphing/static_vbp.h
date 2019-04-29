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
 * @todo Documentation.
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
#include <sstream>

namespace morphstore {
    
    // The vertical bit packed format with a static bit width.
    template<unsigned t_bw, unsigned t_step>
    struct static_vbp_f : public format {
        static_assert(
                (1 <= t_bw) && (t_bw <= std::numeric_limits<uint64_t>::digits),
                "static_vbp: template parameter t_bw must satisfy 1 <= t_bw <= 64"
        );
        static_assert(
                t_step > 0,
                "static_vbp: template parameter t_step must be greater than 0"
        );
        
        static void check_count_values(size_t p_CountValues) {
            // @todo Support arbitrary numbers of data elements.
            const size_t bitsPerReg = t_step * sizeof(uint64_t) * bitsPerByte;
            if(p_CountValues % bitsPerReg) {
                std::stringstream s;
                s
                        << "static_vbp_f: the number of data elements ("
                        << p_CountValues << ") must be a mutliple of the "
                           "number of bits per (vector-)register ("
                        << bitsPerReg << ')';
                throw std::runtime_error(s.str());
            }
        }
        
        static size_t get_size_max_byte(size_t p_CountValues) {
            check_count_values(p_CountValues);
            return p_CountValues * t_bw / bitsPerByte;
        }
    };
    
    /**
     * @brief Morph-operator for the compression to the vertical bit-packed
     * layout with a static bit width.
     * 
     * This operator is completely generic with respect to the configuration of
     * its template parameters. However, invalid combinations will lack a
     * template specialization of the `pack`-function and can, thus, be
     * detected at compile-time.
     */
    template<
            processing_style_t t_ps,
            unsigned t_bw,
            unsigned t_step
    >
    struct morph_t<
            t_ps,
            static_vbp_f<t_bw, t_step>,
            uncompr_f
    > {
        using out_f = static_vbp_f<t_bw, t_step>;
        using in_f = uncompr_f;
        
        static
        const column<out_f> *
        apply(const column<in_f> * inCol) {
            const size_t count64 = inCol->get_count_values();
            out_f::check_count_values(count64);
            const uint8_t * in8 = inCol->get_data();
            
            auto outCol = new column<out_f>(out_f::get_size_max_byte(count64));
            uint8_t * out8 = outCol->get_data();
            const uint8_t * const initOut8 = out8;

            pack<t_ps, t_bw, t_step>(in8, count64, out8);

            outCol->set_meta_data(count64, out8 - initOut8);
            
            return outCol;
        }
    };
    
    /**
     * @brief Morph-operator for the decompression from the vertical bit-packed
     * layout with a static bit width.
     * 
     * This operator is completely generic with respect to the configuration of
     * its template parameters. However, invalid combinations will lack a
     * template specialization of the `unpack`-function and can, thus, be
     * detected at compile-time.
     */
    template<
            processing_style_t t_ps,
            unsigned t_bw,
            unsigned t_step
    >
    struct morph_t<
            t_ps,
            uncompr_f,
            static_vbp_f<t_bw, t_step>
    > {
        using out_f = uncompr_f;
        using in_f = static_vbp_f<t_bw, t_step>;
        
        static
        const column<out_f> *
        apply(const column<in_f> * inCol) {
            const size_t count64 = inCol->get_count_values();
            const uint8_t * in8 = inCol->get_data();
            
            auto outCol = new column<out_f>(out_f::get_size_max_byte(count64));
            uint8_t * out8 = outCol->get_data();
            const uint8_t * const initOut8 = out8;

            unpack<t_ps, t_bw, t_step>(in8, out8, count64);
            
            outCol->set_meta_data(count64, out8 - initOut8);

            return outCol;
        }
    };
    
}
#endif //MORPHSTORE_CORE_MORPHING_STATIC_VBP_H
