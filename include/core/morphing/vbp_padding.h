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
 * @file vbp_padding.h
 * @brief The vertical bit-packed layout with padding in each 64-bit word.
 * 
 * In this particular variant, one packed code word can never span across two
 * memory words.
 * 
 * @todo Documentation.
 */

#ifndef MORPHSTORE_CORE_MORPHING_VBP_PADDING_H
#define MORPHSTORE_CORE_MORPHING_VBP_PADDING_H

#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/morphing/vbp_commons.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_primitives.h>

#include <limits>

#include <cstdint>

namespace morphstore {
    
    // ************************************************************************
    // Layout
    // ************************************************************************
    
    template<unsigned t_Bw, unsigned t_Step>
    struct vbp_padding_l : public layout {
        // @todo Code duplication: Move this into a common base class.
        static_assert(
                (1 <= t_Bw) && (t_Bw <= std::numeric_limits<uint64_t>::digits),
                "vbp_padding_l: template parameter t_Bw must satisfy 1 <= t_Bw <= 64"
        );
        static_assert(
                t_Step > 0,
                "vbp_paddingl: template parameter t_Step must be greater than 0"
        );

        // Assumes that the provided number is a multiple of m_BlockSize.
        static size_t get_size_max_byte(size_t p_CountValues) {
            return p_CountValues / m_BlockSize * t_Step * sizeof(uint64_t) * 10;
        }
        
        // @todo Think about this again.
        static const size_t m_BlockSize =
                std::numeric_limits<uint64_t>::digits / t_Bw * t_Step;
    };
    
    
    
    // ************************************************************************
    // Morph-operators (batch-level)
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    
    template<class t_vector_extension, unsigned t_Bw>
    class morph_batch_t<
            t_vector_extension,
            vbp_padding_l<
                    t_Bw,
                    t_vector_extension::vector_helper_t::element_count::value
            >,
            uncompr_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        using dst_l =vbp_padding_l<
                t_Bw, t_vector_extension::vector_helper_t::element_count::value
        >;
        
        // @todo We could use unrolled routines as we did with vbp_l.
        
    public:
#ifdef VBP_FORCE_INLINE_PACK
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
#endif
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countInLog
        ) {
            using namespace vectorlib;
            
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            base_t * outBase = reinterpret_cast<base_t *>(out8);
            
            const size_t blockSizeVec =
                    dst_l::m_BlockSize / vector_element_count::value;
            
            for(size_t i = 0; i < countInLog; i += dst_l::m_BlockSize) {
                vector_t tmp = load<
                        t_ve, iov::ALIGNED, vector_size_bit::value
                >(inBase);
                inBase += vector_element_count::value;
                for(size_t k = 1; k < blockSizeVec; k++) {
                    tmp = bitwise_or<t_ve>(
                            tmp,
                            shift_left<t_ve>::apply(
                                    load<
                                            t_ve,
                                            iov::ALIGNED,
                                            vector_size_bit::value
                                    >(inBase),
                                    k * t_Bw
                            )
                    );
                    inBase += vector_element_count::value;
                }
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        outBase, tmp
                );
                outBase += vector_element_count::value;
            }
            
            in8 = reinterpret_cast<const uint8_t *>(inBase);
            out8 = reinterpret_cast<uint8_t *>(outBase);
        }
    };
    
    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    
    template<class t_vector_extension, unsigned t_Bw>
    class morph_batch_t<
            t_vector_extension,
            uncompr_f,
            vbp_padding_l<
                    t_Bw,
                    t_vector_extension::vector_helper_t::element_count::value
            >
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        using src_l = vbp_padding_l<
                t_Bw, t_vector_extension::vector_helper_t::element_count::value
        >;
        
        // @todo It would be nice to initialize this in-class. However, the
        // compiler complains because set1 is not constexpr, even when it is
        // defined so.
        static const vector_t mask; // = vectorlib::set1<t_ve, vector_base_t_granularity::value>(bitwidth_max<base_t>(t_bw));
        
        // @todo We could use unrolled routines as we did with vbp_l.
        
    public:
#ifdef VBP_FORCE_INLINE_UNPACK
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
#endif
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
            using namespace vectorlib;
            
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            base_t * outBase = reinterpret_cast<base_t *>(out8);
            
            const size_t blockSizeVec =
                    src_l::m_BlockSize / vector_element_count::value;
            
            for(size_t i = 0; i < countLog; i += src_l::m_BlockSize) {
                const vector_t tmp = load<
                        t_ve, iov::ALIGNED, vector_size_bit::value
                >(inBase);
                inBase += vector_element_count::value;
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        outBase, bitwise_and<t_ve>(mask, tmp)
                );
                outBase += vector_element_count::value;
                for(size_t k = 1; k < blockSizeVec; k++) {
                    store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                            outBase,
                            // @todo The last one does not need to be masked.
                            bitwise_and<t_ve>(
                                    mask,
                                    shift_right<t_ve>::apply(tmp, k * t_Bw)
                            )
                    );
                    outBase += vector_element_count::value;
                }
            }
            
            in8 = reinterpret_cast<const uint8_t *>(inBase);
            out8 = reinterpret_cast<uint8_t *>(outBase);
        }
    };
    
    template<class t_vector_extension, unsigned t_Bw>
    const typename t_vector_extension::vector_t morph_batch_t<
            t_vector_extension,
            uncompr_f,
            vbp_padding_l<
                    t_Bw,
                    t_vector_extension::vector_helper_t::element_count::value
            >
    >::mask = vectorlib::set1<
            t_vector_extension,
            t_vector_extension::vector_helper_t::granularity::value
    >(
            bitwidth_max<typename t_vector_extension::base_t>(t_Bw)
    );
    
}

#endif //MORPHSTORE_CORE_MORPHING_VBP_PADDING_H