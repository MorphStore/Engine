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
#include <core/utils/preprocessor.h>

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
    
#if 1
    // Iterator implementation with a load in each call of next().
    // Seems to be faster for all bit widths.
    
    // @todo This does not work for bit widths 59, 61, 62, 63. But these are not so
    // important anyway.
    // @todo This is probably hard to vectorize.
    template<unsigned t_bw>
    class read_iterator<static_vbp_f<t_bw, 1> > {
        const uint8_t * const m_Data8;
        uint64_t m_Bitpos;

        static const uint64_t m_Mask = bitwidth_max<uint64_t>(t_bw);

    public:
        read_iterator(const uint8_t * p_Data8)
        : m_Data8(p_Data8), m_Bitpos(0) {
            //
        }

        MSV_CXX_ATTRIBUTE_FORCE_INLINE uint64_t next() {
            const uint64_t retVal = ((*reinterpret_cast<const uint64_t *>(m_Data8 + (m_Bitpos >> 3))) >> (m_Bitpos & 0b111)) & m_Mask;
            m_Bitpos += t_bw;
            return retVal;
        };
    };
#else
    // Iterator implementation with a check in each call of next().
    // Seems to be slower for all bit widths.
    
    template<unsigned t_bw>
    class read_iterator<static_vbp_f<t_bw, 1> > {
        const uint64_t * in64;
        uint64_t nextOut;
        uint64_t bitpos;
        uint64_t tmp;

        static const size_t bitsPerWord = std::numeric_limits<uint64_t>::digits;
        static const uint64_t mask = bitwidth_max<uint64_t>(t_bw);

    public:
        read_iterator(const uint64_t * in64) {
            this->in64 = in64;
            nextOut = 0;
            bitpos = bitsPerWord + t_bw;
        }

        MSV_CXX_ATTRIBUTE_FORCE_INLINE uint64_t next() {
            if(MSV_CXX_ATTRIBUTE_UNLIKELY(bitpos == bitsPerWord + t_bw)) {
                tmp = *in64++;
                nextOut = mask & tmp;
                bitpos = t_bw;
            }
            else if(MSV_CXX_ATTRIBUTE_UNLIKELY(bitpos > bitsPerWord && bitpos < bitsPerWord + t_bw)) {
                tmp = *(in64)++;
                nextOut = mask & ((tmp << (bitsPerWord - bitpos + t_bw)) | nextOut);
                bitpos = bitpos - bitsPerWord;
            }
            const uint64_t retVal = nextOut;
            nextOut = mask & (tmp >> bitpos);
            bitpos += t_bw;
            return retVal;
        };
    };
#endif
    
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
            class t_vector_extension,
            unsigned t_bw,
            unsigned t_step
    >
    struct morph_t<
            t_vector_extension,
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

            pack<t_vector_extension, t_bw, t_step>(in8, count64, out8);

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
            class t_vector_extension,
            unsigned t_bw,
            unsigned t_step
    >
    struct morph_t<
            t_vector_extension,
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

            unpack<t_vector_extension, t_bw, t_step>(in8, out8, count64);
            
            outCol->set_meta_data(count64, out8 - initOut8);

            return outCol;
        }
    };
    
}
#endif //MORPHSTORE_CORE_MORPHING_STATIC_VBP_H
