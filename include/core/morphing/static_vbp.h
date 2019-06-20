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
 * bit width for an entire column and facilities for using this format.
 * 
 * This file contains:
 * - the definition of the compressed format
 * - morph-operators for compression and decompression
 * - implementations for accessing the compressed data
 * @todo Documentation.
 */

#ifndef MORPHSTORE_CORE_MORPHING_STATIC_VBP_H
#define MORPHSTORE_CORE_MORPHING_STATIC_VBP_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/morphing/vbp_routines.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/general_vector.h>
#include <vector/primitives/create.h>
#include <vector/primitives/io.h>
// @todo The following includes should not be necessary.
#include <vector/scalar/primitives/compare_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/simd/sse/primitives/compare_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>

#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>

#include <cstdint>
#include <cstring>

namespace morphstore {
    
    // ************************************************************************
    // Format
    // ************************************************************************
    
    /**
     * @brief The vertical bit packed format with a static bit width.
     */
    template<unsigned t_bw, unsigned t_step>
    struct static_vbp_f : public format {
        static_assert(
                (1 <= t_bw) && (t_bw <= std::numeric_limits<uint64_t>::digits),
                "static_vbp_f: template parameter t_bw must satisfy 1 <= t_bw <= 64"
        );
        static_assert(
                t_step > 0,
                "static_vbp_f: template parameter t_step must be greater than 0"
        );

        // Assumes that the provided number is a multiple of m_BlockSize.
        static size_t get_size_max_byte(size_t p_CountValues) {
            return p_CountValues * t_bw / bitsPerByte;
        }
        
        static const size_t m_BlockSize = t_step * sizeof(uint64_t) * bitsPerByte;
    };
    
    
    // ************************************************************************
    // Morph-operators
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    
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
            const size_t countLog = inCol->get_count_values();
            const size_t outCountLogCompr = round_down_to_multiple(
                    countLog, out_f::m_BlockSize
            );
            const size_t outSizeRestByte = uncompr_f::get_size_max_byte(
                    countLog - outCountLogCompr
            );
            
            const uint8_t * in8 = inCol->get_data();
            
            auto outCol = new column<out_f>(
                    get_size_max_byte_any_len<out_f>(countLog)
            );
            uint8_t * out8 = outCol->get_data();
            const uint8_t * const initOut8 = out8;

            pack<t_vector_extension, t_bw, t_step>(in8, outCountLogCompr, out8);
            const size_t sizeComprByte = out8 - initOut8;
            
            if(outSizeRestByte) {
                out8 = create_aligned_ptr(out8);
                memcpy(out8, in8, outSizeRestByte);
            }

            outCol->set_meta_data(
                    countLog, out8 - initOut8 + outSizeRestByte, sizeComprByte
            );
            
            return outCol;
        }
    };
    
    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    
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
            const uint8_t * in8 = inCol->get_data();
            
            const size_t countLog = inCol->get_count_values();
            const uint8_t * const inRest8 = create_aligned_ptr(
                    in8 + inCol->get_size_compr_byte()
            );
            const size_t inCountLogRest = convert_size<uint8_t, uint64_t>(
                    inCol->get_size_used_byte() - (inRest8 - in8)
            );
            const size_t inSizeRestByte = uncompr_f::get_size_max_byte(
                    inCountLogRest
            );
            const size_t inCountLogCompr = countLog - inCountLogRest;
            
            const size_t outSizeByte = out_f::get_size_max_byte(countLog);
            auto outCol = new column<out_f>(outSizeByte);
            uint8_t * out8 = outCol->get_data();

            unpack<t_vector_extension, t_bw, t_step>(
                    in8, out8, inCountLogCompr
            );
            
            memcpy(out8, inRest8, inSizeRestByte);
            
            outCol->set_meta_data(countLog, outSizeByte);

            return outCol;
        }
    };
    
    
    // ************************************************************************
    // Interfaces for accessing compressed data
    // ************************************************************************

    // ------------------------------------------------------------------------
    // Sequential read
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            template<class /*t_vector_extension*/> class t_op_processing_unit,
            unsigned t_bw,
            unsigned t_step
    >
    struct decompress_and_process_batch<
            t_vector_extension,
            static_vbp_f<t_bw, t_step>,
            t_op_processing_unit
    > {
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountIn8,
                typename t_op_processing_unit<t_vector_extension>::state_t & p_State
        ) {
            unpack_and_process<
                    t_vector_extension, t_bw, t_step, t_op_processing_unit
            >(
                    p_In8, p_CountIn8, p_State
            );
        }
    };
    
    // This is deprecated, we decided not to use this approach.
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
    
    // ------------------------------------------------------------------------
    // Sequential write
    // ------------------------------------------------------------------------

    // @todo Take t_step into account correctly.
    template<class t_vector_extension, unsigned t_bw, unsigned t_step>
    class write_iterator<
            t_vector_extension, static_vbp_f<t_bw, t_step>
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        uint8_t * m_Out;
        // @todo Think about this number.
        static const size_t m_CountBuffer = vector_size_bit::value * 16;
        MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value) base_t m_StartBuffer[m_CountBuffer + vector_element_count::value - 1];
        base_t * m_Buffer;
        base_t * const m_EndBuffer;
        size_t m_Count;
        
    public:
        write_iterator(uint8_t * p_Out) :
                m_Out(p_Out),
                m_Buffer(m_StartBuffer),
                m_EndBuffer(m_StartBuffer + m_CountBuffer),
                m_Count(0)
        {
            //
        }
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        void write(vector_t p_Data, vector_mask_t p_Mask) {
            vector::compressstore<
                    t_ve,
                    vector::iov::UNALIGNED, 
                    vector_base_t_granularity::value
            >(m_Buffer, p_Data, p_Mask);
            m_Buffer += vector::count_matches<t_ve>::apply(p_Mask);
            if(MSV_CXX_ATTRIBUTE_UNLIKELY(m_Buffer >= m_EndBuffer)) {
                const uint8_t * buffer8 = reinterpret_cast<uint8_t *>(
                        m_StartBuffer
                );
                // @todo This should not be inlined.
                pack<t_ve, t_bw, t_step>(
                        buffer8, m_CountBuffer, m_Out
                );
                size_t overflow = m_Buffer - m_EndBuffer;
                memcpy(m_StartBuffer, m_EndBuffer, overflow * sizeof(base_t));
                m_Buffer = m_StartBuffer + overflow;
                m_Count += m_CountBuffer;
            }
        }
        
        void done() {
            if(m_Buffer != m_StartBuffer)
                // @todo Error message.
                throw std::runtime_error("ohoh " + std::to_string(m_Buffer - m_StartBuffer));
        }
        
        size_t get_count() const {
            return m_Count;
        }
    };
    
}
#endif //MORPHSTORE_CORE_MORPHING_STATIC_VBP_H
