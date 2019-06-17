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

#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <string>
    

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
        
        static void check_count_values(size_t p_CountValues) {
            // @todo Support arbitrary numbers of data elements.
            if(p_CountValues % m_PageSize64)
                throw std::runtime_error(
                        "dynamic_vbp_f: the number of data elements must be a "
                        "multiple of the page size in data elements"
                );
        }
        
        static size_t get_size_max_byte(size_t p_CountValues) {
            check_count_values(p_CountValues);
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

            const size_t inCountLog = inCol->get_count_values();
            out_f::check_count_values(inCountLog);
            const size_t inCountBase = convert_size<uint8_t, base_t>(
                    inCol->get_size_used_byte()
            );

            const base_t * inBase = inCol->get_data();
            const base_t * const endInBase = inBase + inCountBase;

            auto outCol = new column<out_f>(
                    out_f::get_size_max_byte(inCountLog)
            );
            uint8_t * out8 = outCol->get_data();
            const uint8_t * const initOut8 = out8;

            // Iterate over all input pages.
            while(inBase < endInBase) {
                uint8_t * const outMeta8 = out8;
                out8 += out_f::m_MetaSize8;
                // Iterate over all blocks in the current input page.
                for(
                        unsigned blockIdx = 0;
                        blockIdx < t_PageSizeBlocks;
                        blockIdx++
                ) {
                    // Determine maximum bit width.
                    vector_t pseudoMaxVec = set1<t_ve, vector_base_t_granularity::value>(0);
                    for(
                            unsigned baseIdx = 0;
                            baseIdx < convert_size<uint64_t, base_t>(
                                    t_BlockSize64
                            );
                            baseIdx += vector_element_count::value
                    )
                        pseudoMaxVec = bitwise_or<t_ve>(
                                pseudoMaxVec,
                                load<t_ve, iov::ALIGNED, vector_size_bit::value>(inBase + baseIdx)
                        );
                    
                    // @todo Use vector::hor-primitive when it exists.
                    MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value) base_t tmp[vector_element_count::value];
                    store<t_ve, iov::ALIGNED, vector_size_bit::value>(tmp, pseudoMaxVec);
                    base_t pseudoMaxBase = 1;
                    for(unsigned i = 0; i < vector_element_count::value; i++)
                        pseudoMaxBase |= tmp[i];
                    
                    const unsigned bw = effective_bitwidth(pseudoMaxBase);

                    // Store the bit width to the meta data.
                    outMeta8[blockIdx] = static_cast<uint8_t>(bw);

                    // Pack the data with that bit width.
                    const uint8_t * in8 = reinterpret_cast<const uint8_t *>(inBase);
                    pack_switch<t_ve, vector_element_count::value>(
                            bw, in8, t_BlockSize64, out8
                    );
                    inBase = reinterpret_cast<const base_t *>(in8);
                }
            }

            outCol->set_meta_data(inCountLog, out8 - initOut8);
            
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
            using namespace vector;

            const size_t inCountLog = inCol->get_count_values();
            const size_t inCount8 = inCol->get_size_used_byte();

            const uint8_t * in8 = inCol->get_data();
            const uint8_t * const endIn8 = in8 + inCount8;

            auto outCol = new column<out_f>(
                    out_f::get_size_max_byte(inCountLog)
            );
            uint8_t * out8 = outCol->get_data();

            // Iterate over all pages in the input.
            while(in8 < endIn8) {
                const uint8_t * const inMeta8 = in8;
                in8 += in_f::m_MetaSize8;
                // Iterate over all blocks in the current input page.
                for(
                        unsigned blockIdx = 0;
                        blockIdx < t_PageSizeBlocks;
                        blockIdx++
                )
                    unpack_switch<t_ve, vector_element_count::value>(
                            inMeta8[blockIdx], in8, out8, t_BlockSize64
                    );
            }

            outCol->set_meta_data(
                    inCountLog,
                    out_f::get_size_max_byte(inCountLog)
            );
            
            return outCol;
        }
    };
    
#undef DYNAMIC_VBP_STATIC_ASSERTS_VECTOR_EXTENSION
}


#endif //MORPHSTORE_CORE_MORPHING_DYNAMIC_VBP_H
