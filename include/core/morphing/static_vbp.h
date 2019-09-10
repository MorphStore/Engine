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
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_primitives.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>
#include <tuple>

#include <cstdint>
#include <cstring>

namespace morphstore {
    
    // ************************************************************************
    // Format
    // ************************************************************************
    
    /**
     * @brief The vertical bit packed format with a static bit width.
     */
    template<class t_layout>
    struct static_vbp_f : public format {
        // Assumes that the provided number is a multiple of m_BlockSize.
        static size_t get_size_max_byte(size_t p_CountValues) {
            return t_layout::get_size_max_byte(p_CountValues);
        }
        
        static const size_t m_BlockSize = t_layout::m_BlockSize;
    };
    
    
    // ************************************************************************
    // Morph-operators (batch-level)
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
     * template specialization of the `morph_batch`-function and can, thus, be
     * detected at compile-time.
     */
    template<class t_vector_extension, class t_layout>
    struct morph_batch_t<
            t_vector_extension, static_vbp_f<t_layout>, uncompr_f
    > {
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
            return morph_batch<t_vector_extension, t_layout, uncompr_f>(
                    in8, out8, countLog
            );
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
     * template specialization of the `morph_batch`-function and can, thus, be
     * detected at compile-time.
     */
    template<class t_vector_extension, class t_layout>
    struct morph_batch_t<
            t_vector_extension, uncompr_f, static_vbp_f<t_layout>
    > {
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
            return morph_batch<t_vector_extension, uncompr_f, t_layout>(
                    in8, out8, countLog
            );
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
            class t_layout,
            template<class, class...> class t_op_vector,
            class ... t_extra_args
    >
    struct decompress_and_process_batch<
            t_vector_extension,
            static_vbp_f<t_layout>,
            t_op_vector,
            t_extra_args ...
    > {
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountIn8,
                typename t_op_vector<
                        t_vector_extension,
                        t_extra_args ...
                >::state_t & p_State
        ) {
            decompress_and_process_batch<
                    t_vector_extension,
                    t_layout,
                    t_op_vector,
                    t_extra_args ...
            >::apply(p_In8, p_CountIn8, p_State);
        }
    };
    
    // This is deprecated, we decided not to use this approach.
    // @todo Remove this.
#if 0
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
#elif 0
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
    // Random read
    // ------------------------------------------------------------------------

    template<class t_vector_extension, class t_layout>
    class random_read_access<t_vector_extension, static_vbp_f<t_layout> > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        typename random_read_access<t_ve, t_layout>::type m_Internal;
        
    public:
        // Alias to itself, in this case.
        using type = random_read_access<t_vector_extension, static_vbp_f<t_layout> >;
        
        random_read_access(const base_t * p_Data) : m_Internal(p_Data) {
            //
        }

        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        vector_t get(const vector_t & p_Positions) {
            return m_Internal.get(p_Positions);
        }
    };
    
    // ------------------------------------------------------------------------
    // Sequential write
    // ------------------------------------------------------------------------

    // @todo Take t_step into account correctly.
    template<class t_vector_extension, class t_layout>
    class selective_write_iterator<
            t_vector_extension, static_vbp_f<t_layout>
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using out_f = static_vbp_f<t_layout>;
        
        uint8_t * m_Out;
        const uint8_t * const m_InitOut;
        // @todo Think about this number.
        static const size_t m_CountBuffer = out_f::m_BlockSize * 16;
        MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value) base_t m_StartBuffer[
                m_CountBuffer + vector_element_count::value - 1
        ];
        base_t * m_Buffer;
        base_t * const m_EndBuffer;
        size_t m_Count;
        
        void compress_buffer() {
            const uint8_t * buffer8 = reinterpret_cast<uint8_t *>(
                    m_StartBuffer
            );
            // @todo This should not be inlined.
            morph_batch<t_ve, t_layout, uncompr_f>(
                    buffer8, m_Out, m_CountBuffer
            );
            size_t overflow = m_Buffer - m_EndBuffer;
            memcpy(m_StartBuffer, m_EndBuffer, overflow * sizeof(base_t));
            m_Buffer = m_StartBuffer + overflow;
            m_Count += m_CountBuffer;
        }
        
    public:
        selective_write_iterator(uint8_t * p_Out) :
                m_Out(p_Out),
                m_InitOut(m_Out),
                m_Buffer(m_StartBuffer),
                m_EndBuffer(m_StartBuffer + m_CountBuffer),
                m_Count(0)
        {
            //
        }
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(
                vector_t p_Data, vector_mask_t p_Mask, uint8_t p_MaskPopCount
        ) {
            vectorlib::compressstore<
                    t_ve,
                    vectorlib::iov::UNALIGNED,
                    vector_base_t_granularity::value
            >(m_Buffer, p_Data, p_Mask);
            m_Buffer += p_MaskPopCount;
            if(MSV_CXX_ATTRIBUTE_UNLIKELY(m_Buffer >= m_EndBuffer))
                compress_buffer();
        }

        MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(
                vector_t p_Data, vector_mask_t p_Mask
        ) {
            write(
                    p_Data,
                    p_Mask,
                    vectorlib::count_matches<t_vector_extension>::apply(p_Mask)
            );
        }
        
        std::tuple<size_t, bool, uint8_t *> done() {
            const size_t countLog = m_Buffer - m_StartBuffer;
            bool startedUncomprPart = false;
            size_t outSizeComprByte;
            if(countLog) {
                const size_t outCountLogCompr = round_down_to_multiple(
                        countLog, out_f::m_BlockSize
                );

                const uint8_t * buffer8 = reinterpret_cast<uint8_t *>(
                        m_StartBuffer
                );
                morph_batch<t_ve, t_layout, uncompr_f>(
                    buffer8, m_Out, outCountLogCompr
                );
                outSizeComprByte = m_Out - m_InitOut;

                const size_t outCountLogRest = countLog - outCountLogCompr;
                if(outCountLogRest) {
                    m_Out = create_aligned_ptr(m_Out);
                    const size_t sizeOutLogRest = uncompr_f::get_size_max_byte(outCountLogRest);
                    memcpy(
                            m_Out,
                            m_StartBuffer + outCountLogCompr,
                            sizeOutLogRest
                    );
                    m_Out += sizeOutLogRest;
                    startedUncomprPart = true;
                }
                
                m_Count += countLog;
            }
            else
                outSizeComprByte = m_Out - m_InitOut;

            return std::make_tuple(
                    outSizeComprByte,
                    startedUncomprPart,
                    m_Out
            );
        }
        
        size_t get_count_values () const {
            return m_Count;
        }
    };
    
}
#endif //MORPHSTORE_CORE_MORPHING_STATIC_VBP_H
