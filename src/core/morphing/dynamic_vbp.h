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
// @todo Remove this once dynamic_vbp_f is generic w.r.t. the underlying layout.
#include <core/morphing/vbp.h>
#include <core/morphing/vbp_commons.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_primitives.h>

#include <limits>
#include <stdexcept>
#include <string>
    
#include <cstdint>
#include <cstring>

#define DYNAMIC_VBP_STATIC_ASSERTS_VECTOR_EXTENSION \
    static_assert( \
            t_BlockSizeLog % vector_size_bit::value == 0, \
            "dynamic_vbp_f: template parameter t_BlockSizeLog must be a multiple of the vector size in bits" \
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
     * - `t_BlockSizeLog` The number of logical data elements per block
     * - `t_PageSizeBlocks` The number of blocks per page.
     * 
     * In fact, for a block size of 128 data elements and 16 blocks per page,
     * this is the format called SIMD-BP128 in the literature, with the
     * following differences:
     * - we use 64-bit instead of 32-bit data elements
     * - we pack a block of zeros with a bit width of 1 instead of omitting it
     */
    // @todo Make this properly depend on the underlying layout.
    template<size_t t_BlockSizeLog, size_t t_PageSizeBlocks, unsigned t_Step>
    struct dynamic_vbp_f : public format {
        static_assert(
                t_BlockSizeLog > 0,
                "dynamic_vbp_f: template parameter t_BlockSizeLog must be greater than 0"
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
                t_BlockSizeLog % t_Step == 0,
                "dynamic_vbp_f: template parameter t_BlockSizeLog must be a multiple of template parameter t_Step"
        );
        
        static const size_t m_PageSizeLog = t_PageSizeBlocks * t_BlockSizeLog;
        static const size_t m_MetaSize8 = t_PageSizeBlocks * sizeof(uint8_t);
        
        // Assumes that the provided number is a multiple of m_BlockSize.
        static size_t get_size_max_byte(size_t p_CountValues) {
            // Actually, the format needs one meta data section of m_MetaSize8
            // bytes for each *page*. However, depending on the block size of
            // cascades, write-iterators, etc., there could be incomplete
            // pages. The worst case is only one block per page, so we reserve
            // enough memory for one meta data section per *block*.
            const size_t blockCount = p_CountValues / t_BlockSizeLog;
            const size_t totalMetaSizeByte = blockCount * m_MetaSize8;
            // These numbers are worst cases, which are only reached if all
            // blocks require the maximum bit width of 64.
            const size_t totalDataSizeByte = p_CountValues * sizeof(uint64_t);
            return totalDataSizeByte + totalMetaSizeByte;
        }
        
        static const size_t m_BlockSize = t_BlockSizeLog;
        
        template<class t_vector_extension>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static unsigned determine_max_bitwidth(
                const typename t_vector_extension::base_t * p_Buffer
        ) {
            using t_ve = t_vector_extension;
            IMPORT_VECTOR_BOILER_PLATE(t_ve)
            
            using namespace vectorlib;
            
            vector_t pseudoMaxVec = set1<
                    t_ve, vector_base_t_granularity::value
            >(0);
            for(
                    unsigned baseIdx = 0;
                    baseIdx < convert_size<uint64_t, base_t>(t_BlockSizeLog);
                    baseIdx += vector_element_count::value
            )
                pseudoMaxVec = bitwise_or<t_ve>(
                        pseudoMaxVec,
                        load<t_ve, iov::ALIGNED, vector_size_bit::value>(
                                p_Buffer + baseIdx
                        )
                );

            // @todo Use vectorlib::hor-primitive when it exists.
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
    // Morph-operators (batch-level)
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            size_t t_BlockSizeLog,
            size_t t_PageSizeBlocks,
            unsigned t_Step
    >
    struct morph_batch_t<
            t_vector_extension,
            dynamic_vbp_f<t_BlockSizeLog, t_PageSizeBlocks, t_Step>,
            uncompr_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
                
        DYNAMIC_VBP_STATIC_ASSERTS_VECTOR_EXTENSION
        
        using out_f = dynamic_vbp_f<t_BlockSizeLog, t_PageSizeBlocks, t_Step>;
        
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
            using namespace vectorlib;
            
            const size_t countLogComplPages = round_down_to_multiple(
                    countLog, out_f::m_PageSizeLog
            );
            const size_t countLogIncomplPage = countLog - countLogComplPages;
            
            const uint8_t * const endInComplPagesBase8 =
                    in8 + convert_size<uint64_t, uint8_t>(countLogComplPages);

            // Iterate over all complete input pages.
            while(in8 < endInComplPagesBase8) {
                uint8_t * const outMeta8 = out8;
                out8 += out_f::m_MetaSize8;
                // Iterate over all blocks in the current input page.
                for(
                        unsigned blockIdx = 0;
                        blockIdx < t_PageSizeBlocks;
                        blockIdx++
                ) {
                    // Determine the maximum bit width.
                    const unsigned bw = out_f::template determine_max_bitwidth<
                            t_ve
                    >(reinterpret_cast<const base_t *>(in8));

                    // Store the bit width to the meta data.
                    outMeta8[blockIdx] = static_cast<uint8_t>(bw);

                    // Pack the data with that bit width.
                    pack_switch<t_ve, vbp_l, vector_element_count::value>(
                            bw, in8, out8, t_BlockSizeLog
                    );
                }
            }
            // Handle the incomplete page at the end, if necessary.
            if(countLogIncomplPage) {
                uint8_t * const outMeta8 = out8;
                out8 += out_f::m_MetaSize8;
                const size_t countBlocksIncomplPage =
                        countLogIncomplPage / t_BlockSizeLog;
                for(
                        unsigned blockIdx = 0;
                        blockIdx < countBlocksIncomplPage;
                        blockIdx++
                ) {
                    // Determine the maximum bit width.
                    const unsigned bw = out_f::template determine_max_bitwidth<
                            t_ve
                    >(reinterpret_cast<const base_t *>(in8));

                    // Store the bit width to the meta data.
                    outMeta8[blockIdx] = static_cast<uint8_t>(bw);

                    // Pack the data with that bit width.
                    pack_switch<t_ve, vbp_l, vector_element_count::value>(
                            bw, in8, out8, t_BlockSizeLog
                    );
                }
                // Fill the meta data of remaining blocks with a marker to
                // indicate that they are unused.
                memset(
                        outMeta8 + countBlocksIncomplPage,
                        VBP_BW_NOBLOCK,
                        out_f::m_MetaSize8 - countBlocksIncomplPage
                );
            }
        }
    };
    
    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            size_t t_BlockSizeLog,
            size_t t_PageSizeBlocks,
            unsigned t_Step
    >
    struct morph_batch_t<
            t_vector_extension,
            uncompr_f,
            dynamic_vbp_f<t_BlockSizeLog, t_PageSizeBlocks, t_Step>
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
                
        DYNAMIC_VBP_STATIC_ASSERTS_VECTOR_EXTENSION
        
        using out_f = uncompr_f;
        using in_f = dynamic_vbp_f<t_BlockSizeLog, t_PageSizeBlocks, t_Step>;
        
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
            // Iterate over all complete pages and possibly the final
            // incomplete page in the input.
            for(size_t countLogDecompr = 0; countLogDecompr < countLog;) {
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
                ) {
                    const uint8_t bw = inMeta8[blockIdx];
                    unpack_switch<t_ve, vbp_l, vector_element_count::value>(
                            bw, in8, out8, t_BlockSizeLog
                    );
                    countLogDecompr += (bw != VBP_BW_NOBLOCK) * t_BlockSizeLog;
                }
            }
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
            size_t t_BlockSizeLog,
            size_t t_PageSizeBlocks,
            unsigned t_Step,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    struct decompress_and_process_batch<
            t_vector_extension,
            dynamic_vbp_f<t_BlockSizeLog, t_PageSizeBlocks, t_Step>,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using in_f = dynamic_vbp_f<t_BlockSizeLog, t_PageSizeBlocks, t_Step>;
        
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountInLog,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & p_State
        ) {
            // Iterate over all complete pages and possibly the final
            // incomplete page in the input.
            for(size_t countLogProcessed = 0; countLogProcessed < p_CountInLog;) {
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
                    const uint8_t bw = inMeta8[blockIdx];
                    decompress_and_process_batch_switch<
                            t_ve,
                            vbp_l,
                            vector_element_count::value,
                            t_op_vector,
                            t_extra_args ...
                    >(
                            bw, p_In8, t_BlockSizeLog, p_State
                    );
                    countLogProcessed += (bw != VBP_BW_NOBLOCK) * t_BlockSizeLog;
                }
            }
        }
    };
    
#undef DYNAMIC_VBP_STATIC_ASSERTS_VECTOR_EXTENSION
}


#endif //MORPHSTORE_CORE_MORPHING_DYNAMIC_VBP_H
