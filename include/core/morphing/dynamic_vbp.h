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
 * @file dynamic_vbp.h
 * @brief A generalization of SIMD-BP128 and facilities for using this format.
 */

#ifndef MORPHSTORE_CORE_MORPHING_DYNAMIC_VBP_H
#define MORPHSTORE_CORE_MORPHING_DYNAMIC_VBP_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/morphing/vbp_routines.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>
#include <vector/primitives/compare.h>
#include <vector/primitives/create.h>
#include <vector/primitives/io.h>
#include <vector/primitives/logic.h>
// @todo The following includes should not be necessary.
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/scalar/primitives/logic_scalar.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/logic_avx2.h>
//#include <vector/simd/avx512/primitives/create_avx512.h>
//#include <vector/simd/avx512/primitives/io_avx512.h>
//#include <vector/simd/avx512/primitives/logic_avx512.h>
#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/logic_sse.h>

#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <string>
    
#include <cstdint>
#include <cstring>

#define DYNAMIC_VBP_STATIC_ASSERTS_VECTOR_EXTENSION \
    static_assert( \
            t_BlockSize64 % vector_size_bit::value == 0, \
            "dynamic_vbp_f: template parameter t_BlockSize64 must be a multiple of the vector size in bits" \
    ); \
    static_assert( \
            t_PageSizeBlocks % vector_size_byte::value == 0, \
            "dynamic_vbp_f: template parameter t_PageSizeBlocks must be a multiple of the vector size in bytes" \
    );

namespace morphstore {
    
    // ************************************************************************
    // Format
    // ************************************************************************
    
    /**
     * @brief A generalization of the compressed format of SIMD-BP128.
     * 
     * This compressed format divides the uncompressed data into blocks of a
     * fixed-size (in terms of data elements). For each block, all data
     * elements in the block are represented in the vertical bit-packed layout
     * with the number of bits required for the largest value in the block,
     * i.e., each block has its individual bit width. These bit widths
     * (meta data) are stored at the page-level, whereby each page consists of
     * a fixed number of blocks.
     * 
     * There are two template parameters controlling the concrete layout:
     * - `t_BlockSize64` The number of uncompressed (64-bit) data elements per
     *   block
     * - `t_PageSizeBlocks` The number of blocks per page.
     * 
     * In fact, for a block size of 128 data elements and 16 blocks per page,
     * this is the format called SIMD-BP128 in the literature, with the 
     * difference that we use 64-bit instead of 32-bit date elements.
     */
    template<size_t t_BlockSize64, size_t t_PageSizeBlocks, unsigned t_Step>
    struct dynamic_vbp_f : public format {
        static_assert(
                t_BlockSize64 > 0,
                "dynamic_vbp_f: template parameter t_BlockSize64 must be greater than 0"
        );
        static_assert(
                t_PageSizeBlocks > 0,
                "dynamic_vbp_f: template parameter t_PageSizeBlocks must be greater than 0"
        );
        static_assert(
                t_Step > 0,
                "dynamic_vbp_f: template parameter t_Step must be greater than 0"
        );
        static_assert(
                t_BlockSize64 % t_Step == 0,
                "dynamic_vbp_f: template parameter t_BlockSize64 must be a multiple of template parameter t_Step"
        );
        
        static const size_t m_PageSize64 = t_PageSizeBlocks * t_BlockSize64;
        static const size_t m_MetaSize8 = t_PageSizeBlocks * sizeof(uint8_t);
        
        // Assumes that the provided number is a multiple of m_BlockSize.
        static size_t get_size_max_byte(size_t p_CountValues) {
            // These numbers are exact (assuming that the check above
            // succeeded).
            const size_t pageCount = p_CountValues / m_PageSize64;
            const size_t totalMetaSizeByte = pageCount * m_MetaSize8;
            // These numbers are worst cases, which are only reached if all
            // blocks require the maximum bit width of 64.
            const size_t totalDataSizeByte = p_CountValues *
                    std::numeric_limits<uint64_t>::digits / bitsPerByte;
            return totalDataSizeByte + totalMetaSizeByte;
        }
        
        static const size_t m_BlockSize = t_BlockSize64;
        
        template<class t_vector_extension>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static unsigned determine_max_bitwidth(
                const typename t_vector_extension::base_t * p_Buffer
        ) {
            using t_ve = t_vector_extension;
            IMPORT_VECTOR_BOILER_PLATE(t_ve)
            
            using namespace vector;
            
            vector_t pseudoMaxVec = set1<
                    t_ve, vector_base_t_granularity::value
            >(0);
            for(
                    unsigned baseIdx = 0;
                    baseIdx < convert_size<uint64_t, base_t>(t_BlockSize64);
                    baseIdx += vector_element_count::value
            )
                pseudoMaxVec = bitwise_or<t_ve>(
                        pseudoMaxVec,
                        load<t_ve, iov::ALIGNED, vector_size_bit::value>(
                                p_Buffer + baseIdx
                        )
                );

            // @todo Use vector::hor-primitive when it exists.
            MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value)
            base_t tmp[vector_element_count::value];
            store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                    tmp, pseudoMaxVec
            );
            base_t pseudoMaxBase = 1;
            for(unsigned i = 0; i < vector_element_count::value; i++)
                pseudoMaxBase |= tmp[i];

            return effective_bitwidth(pseudoMaxBase);
        }
    };
    
    
    // ************************************************************************
    // Morph-operators
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            size_t t_BlockSize64,
            size_t t_PageSizeBlocks,
            unsigned t_Step
    >
    struct morph_t<
            t_vector_extension,
            dynamic_vbp_f<t_BlockSize64, t_PageSizeBlocks, t_Step>,
            uncompr_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
                
        DYNAMIC_VBP_STATIC_ASSERTS_VECTOR_EXTENSION
        
        using out_f = dynamic_vbp_f<t_BlockSize64, t_PageSizeBlocks, t_Step>;
        using in_f = uncompr_f;
        
        static
        const column<out_f> *
        apply(const column<in_f> * inCol) {
            using namespace vector;
            
            const size_t countLog = inCol->get_count_values();
            const size_t outCountLogCompr = round_down_to_multiple(
                    countLog, out_f::m_BlockSize
            );
            const size_t outCountLogComprComplPages = round_down_to_multiple(
                    outCountLogCompr, out_f::m_PageSize64
            );
            const size_t outCountLogComprIncomplPage =
                    outCountLogCompr - outCountLogComprComplPages;
            const size_t outSizeRestByte = uncompr_f::get_size_max_byte(
                    countLog - outCountLogCompr
            );
            
            const base_t * inBase = inCol->get_data();
            const base_t * const endInComprComplPagesBase =
                    inBase + outCountLogComprComplPages;

            auto outCol = new column<out_f>(
                    get_size_max_byte_any_len<out_f>(countLog)
            );
            uint8_t * out8 = outCol->get_data();
            const uint8_t * const initOut8 = out8;

            // Iterate over all complete input pages.
            while(inBase < endInComprComplPagesBase) {
                uint8_t * const outMeta8 = out8;
                out8 += out_f::m_MetaSize8;
                // Iterate over all blocks in the current input page.
                for(
                        unsigned blockIdx = 0;
                        blockIdx < t_PageSizeBlocks;
                        blockIdx++
                ) {
                    // Determine maximum bit width.
                    const unsigned bw = out_f::template determine_max_bitwidth<
                            t_ve
                    >(inBase);

                    // Store the bit width to the meta data.
                    outMeta8[blockIdx] = static_cast<uint8_t>(bw);

                    // Pack the data with that bit width.
                    const uint8_t * in8 = reinterpret_cast<const uint8_t *>(
                            inBase
                    );
                    pack_switch<t_ve, vector_element_count::value>(
                            bw, in8, t_BlockSize64, out8
                    );
                    inBase = reinterpret_cast<const base_t *>(in8);
                }
            }
            // Handle the incomplete page at the end, if necessary.
            if(outCountLogComprIncomplPage) {
                uint8_t * const outMeta8 = out8;
                out8 += out_f::m_MetaSize8;
                const size_t countBlocksIncomplPage =
                        outCountLogComprIncomplPage / t_BlockSize64;
                for(
                        unsigned blockIdx = 0;
                        blockIdx < countBlocksIncomplPage;
                        blockIdx++
                ) {
                    // Determine maximum bit width.
                    const unsigned bw = out_f::template determine_max_bitwidth<
                            t_ve
                    >(inBase);

                    // Store the bit width to the meta data.
                    outMeta8[blockIdx] = static_cast<uint8_t>(bw);

                    // Pack the data with that bit width.
                    const uint8_t * in8 = reinterpret_cast<const uint8_t *>(
                            inBase
                    );
                    pack_switch<t_ve, vector_element_count::value>(
                            bw, in8, t_BlockSize64, out8
                    );
                    inBase = reinterpret_cast<const base_t *>(in8);
                }
                // Fill the meta data of remaining blocks with a marker to
                // indicate that they are unused.
                memset(
                        outMeta8 + countBlocksIncomplPage,
                        VBP_BW_NOBLOCK,
                        out_f::m_MetaSize8 - countBlocksIncomplPage
                );
            }
            const size_t sizeComprByte = out8 - initOut8;
            
            if(outSizeRestByte) {
                out8 = create_aligned_ptr(out8);
                memcpy(out8, inBase, outSizeRestByte);
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
    
    template<
            class t_vector_extension,
            size_t t_BlockSize64,
            size_t t_PageSizeBlocks,
            unsigned t_Step
    >
    struct morph_t<
            t_vector_extension,
            uncompr_f,
            dynamic_vbp_f<t_BlockSize64, t_PageSizeBlocks, t_Step>
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
                
        DYNAMIC_VBP_STATIC_ASSERTS_VECTOR_EXTENSION
        
        using out_f = uncompr_f;
        using in_f = dynamic_vbp_f<t_BlockSize64, t_PageSizeBlocks, t_Step>;
        
        static
        const column<out_f> *
        apply(const column<in_f> * inCol) {
            const uint8_t * in8 = inCol->get_data();
            const uint8_t * const initIn8 = in8;
            const size_t inSizeComprByte = inCol->get_size_compr_byte();
            const size_t inSizeUsedByte = inCol->get_size_used_byte();
            const uint8_t * const endInCompr8 = in8 + inSizeComprByte;
            
            const size_t countLog = inCol->get_count_values();
            
            auto outCol = new column<out_f>(
                    out_f::get_size_max_byte(countLog)
            );
            uint8_t * out8 = outCol->get_data();

            // Iterate over all complete pages and possibly the final
            // incomplete page in the input.
            while(in8 < endInCompr8) {
                const uint8_t * const inMeta8 = in8;
                in8 += in_f::m_MetaSize8;
                // Iterate over all blocks in the current input page. In the
                // final incomplete page, a block could be non-existent, which
                // is marked by a bit width of VBP_BW_NOBLOCK and handled in
                // unpack_switch.
                for(
                        unsigned blockIdx = 0;
                        blockIdx < t_PageSizeBlocks;
                        blockIdx++
                )
                    unpack_switch<t_ve, vector_element_count::value>(
                            inMeta8[blockIdx], in8, out8, t_BlockSize64
                    );
            }
            
            if(inSizeComprByte < inSizeUsedByte) {
                // If the input column has an uncompressed rest part.
                const uint8_t * const inRest8 = create_aligned_ptr(endInCompr8);
                const size_t inCountLogRest = convert_size<uint8_t, uint64_t>(
                        inSizeUsedByte - (inRest8 - initIn8)
                );
                const size_t inSizeRestByte = uncompr_f::get_size_max_byte(
                        inCountLogRest
                );
                memcpy(out8, inRest8, inSizeRestByte);
            }

            outCol->set_meta_data(
                    countLog, out_f::get_size_max_byte(countLog)
            );
            
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
            size_t t_BlockSize64,
            size_t t_PageSizeBlocks,
            unsigned t_Step,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    struct decompress_and_process_batch<
            t_vector_extension,
            dynamic_vbp_f<t_BlockSize64, t_PageSizeBlocks, t_Step>,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using in_f = dynamic_vbp_f<t_BlockSize64, t_PageSizeBlocks, t_Step>;
        
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountIn8,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & p_State
        ) {
            const uint8_t * const endIn8 = p_In8 + p_CountIn8;
            
            // Iterate over all complete pages and possibly the final
            // incomplete page in the input.
            while(p_In8 < endIn8) {
                const uint8_t * const inMeta8 = p_In8;
                p_In8 += in_f::m_MetaSize8;
                // Iterate over all blocks in the current input page. In the
                // final incomplete page, a block could be non-existent, which
                // is marked by a bit width of VBP_BW_NOBLOCK and handled in
                // unpack_switch.
                for(
                        unsigned blockIdx = 0;
                        blockIdx < t_PageSizeBlocks;
                        blockIdx++
                ) {
                    const unsigned bw = inMeta8[blockIdx];
                    const size_t blockSize8 = t_BlockSize64 * bw / bitsPerByte;
                    unpack_and_process_switch<
                            t_ve,
                            vector_element_count::value,
                            t_op_vector,
                            t_extra_args ...
                    >(
                            bw, p_In8, blockSize8, p_State
                    );
                }
            }
        }
    };
    
    // ------------------------------------------------------------------------
    // Sequential write
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            size_t t_BlockSize64,
            size_t t_PageSizeBlocks,
            unsigned t_Step
    >
    class selective_write_iterator<
            t_vector_extension,
            dynamic_vbp_f<t_BlockSize64, t_PageSizeBlocks, t_Step>
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using out_f = dynamic_vbp_f<t_BlockSize64, t_PageSizeBlocks, t_Step>;
        
        uint8_t * m_OutMeta;
        uint8_t * m_OutData;
        size_t m_BlockIdxInPage;
        const uint8_t * const m_InitOut;
        // @todo Think about this number.
        static const size_t m_CountBuffer = out_f::m_PageSize64;
        static const size_t m_CountBufferBlocks = m_CountBuffer / t_BlockSize64;
        MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value) base_t m_StartBuffer[
                m_CountBuffer + vector_element_count::value - 1
        ];
        base_t * m_Buffer;
        base_t * const m_EndBuffer;
        size_t m_Count;
        
        void compress_buffer() {
            // Assumes that m_CountBuffer is 2^i * out_f::m_PageSize64,
            // where i <= 0, i.e., the buffer is as large as a page or a
            // half of it, our a quarter of it, ....
            const uint8_t * buffer8 = reinterpret_cast<const uint8_t *>(
                    m_StartBuffer
            );
            for(
                    unsigned blockIdx = 0;
                    blockIdx < m_CountBufferBlocks;
                    blockIdx++
            ) {
                    const unsigned bw = out_f::template determine_max_bitwidth<
                            t_ve
                    >(reinterpret_cast<const base_t *>(buffer8));
                m_OutMeta[m_BlockIdxInPage + blockIdx] =
                        static_cast<uint8_t>(bw);
                pack_switch<t_ve, t_Step>(
                        bw, buffer8, t_BlockSize64, m_OutData
                );
            }
            m_BlockIdxInPage += m_CountBufferBlocks;
            if(m_BlockIdxInPage == t_PageSizeBlocks) {
                m_OutMeta = m_OutData;
                m_OutData += out_f::m_MetaSize8;
                m_BlockIdxInPage = 0;
            }

            size_t overflow = m_Buffer - m_EndBuffer;
            memcpy(m_StartBuffer, m_EndBuffer, overflow * sizeof(base_t));
            m_Buffer = m_StartBuffer + overflow;
            m_Count += m_CountBuffer;
        }

    public:
        selective_write_iterator(uint8_t * p_Out) :
                m_OutMeta(p_Out),
                m_OutData(m_OutMeta + out_f::m_MetaSize8),
                m_BlockIdxInPage(0),
                m_InitOut(p_Out),
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
            if(MSV_CXX_ATTRIBUTE_UNLIKELY(m_Buffer >= m_EndBuffer))
                compress_buffer();
        }

        std::tuple<size_t, bool, uint8_t *> done() {
            const size_t countLog = m_Buffer - m_StartBuffer;
            bool startedUncomprPart = false;
            size_t outSizeComprByte;
            uint8_t * endOut;
            if(countLog) {
                const size_t outCountLogCompr = round_down_to_multiple(
                        countLog, t_BlockSize64
                );

                // Assumes that m_CountBuffer is 2^i * out_f::m_PageSize64,
                // where i <= 0, i.e., the buffer is as large as a page or a
                // half of it, our a quarter of it, ....
                const size_t countFullBlocks = countLog / t_BlockSize64;
                const uint8_t * buffer8 = reinterpret_cast<const uint8_t *>(
                        m_StartBuffer
                );
                for(
                        unsigned blockIdx = 0;
                        blockIdx < countFullBlocks;
                        blockIdx++
                ) {
                    const unsigned bw = out_f::template determine_max_bitwidth<
                            t_ve
                    >(reinterpret_cast<const base_t *>(buffer8));
                    m_OutMeta[m_BlockIdxInPage + blockIdx] =
                            static_cast<uint8_t>(bw);
                    pack_switch<t_ve, t_Step>(
                            bw, buffer8, t_BlockSize64, m_OutData
                    );
                }
                m_BlockIdxInPage += countFullBlocks;
                // Fill the meta data of remaining blocks with a marker to
                // indicate that they are unused.
                memset(
                        m_OutMeta + m_BlockIdxInPage,
                        VBP_BW_NOBLOCK,
                        out_f::m_MetaSize8 - m_BlockIdxInPage
                );
                endOut = m_BlockIdxInPage ? m_OutData : m_OutMeta;
                outSizeComprByte = endOut - m_InitOut;

                const size_t outCountLogRest = countLog - outCountLogCompr;
                if(outCountLogRest) {
                    endOut = create_aligned_ptr(endOut);
                    const size_t sizeOutLogRest =
                            uncompr_f::get_size_max_byte(outCountLogRest);
                    memcpy(
                            endOut,
                            m_StartBuffer + outCountLogCompr,
                            sizeOutLogRest
                    );
                    endOut += sizeOutLogRest;
                    startedUncomprPart = true;
                }
                
                m_Count += countLog;
            }
            else {
                endOut = m_BlockIdxInPage ? m_OutData : m_OutMeta;
                outSizeComprByte = endOut - m_InitOut;
            }

            return std::make_tuple(
                    outSizeComprByte, startedUncomprPart, endOut
            );
        }

        size_t get_count_values() const {
            return m_Count;
        }
    };
    
#undef DYNAMIC_VBP_STATIC_ASSERTS_VECTOR_EXTENSION
}


#endif //MORPHSTORE_CORE_MORPHING_DYNAMIC_VBP_H
