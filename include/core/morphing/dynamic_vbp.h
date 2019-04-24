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
 * @brief Format struct and (de)compression morph operators for a
 * generalization of the compressed format of SIMD-BP128.
 */

#ifndef MORPHSTORE_CORE_MORPHING_DYNAMIC_VBP_H
#define MORPHSTORE_CORE_MORPHING_DYNAMIC_VBP_H

#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/morphing/vbp_routines.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>

#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace morphstore {
    
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
    template<size_t t_BlockSize64, size_t t_PageSizeBlocks>
    struct dynamic_vbp_f : public format {
        static const size_t m_PageSize64 = t_PageSizeBlocks * t_BlockSize64;
        static const size_t m_MetaSize8 = t_PageSizeBlocks * sizeof(uint8_t);
        
        template<typename t_vec_t>
        static void check_compatibility() {
            if(t_BlockSize64 % (sizeof(t_vec_t) * bitsPerByte))
                throw std::runtime_error(
                        "dynamic_vbp_f: the number of data elements per block "
                        "must be a multiple of the number of bits per vector "
                        "register"
                );
            if(t_PageSizeBlocks % sizeof(t_vec_t))
                throw std::runtime_error(
                        "dynamic_vbp_f: the number of blocks per page must be "
                        "a multiple of the number of bytes per vector register"
                );
        }
        
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
    
    template<size_t t_BlockSize64, size_t t_PageSizeBlocks>
    struct morph_t<
            processing_style_t::vec128,
            dynamic_vbp_f<t_BlockSize64, t_PageSizeBlocks>,
            uncompr_f
    > {
        using out_f = dynamic_vbp_f<t_BlockSize64, t_PageSizeBlocks>;
        using in_f = uncompr_f;
        
        static
        const column<out_f> *
        apply(const column<in_f> * inCol) {
            out_f::template check_compatibility<__m128i>();

            const size_t inCount64 = inCol->get_count_values();
            out_f::check_count_values(inCount64);
            const size_t inCount128 = convert_size<uint64_t, __m128i>(
                    inCount64
            );

            const __m128i * in128 = inCol->get_data();
            const __m128i * const endIn128 = in128 + inCount128;

            auto outCol = new column<out_f>(
                    out_f::get_size_max_byte(inCount64)
            );
            __m128i * out128 = outCol->get_data();
            const __m128i * const initOut128 = out128;

            // Iterate over all input pages.
            while(in128 < endIn128) {
                uint8_t * const outMeta8 = reinterpret_cast<uint8_t *>(out128);
                out128 += convert_size<uint8_t, __m128i>(out_f::m_MetaSize8);
                // Iterate over all blocks in the current input page.
                for(
                        unsigned blockIdx = 0;
                        blockIdx < t_PageSizeBlocks;
                        blockIdx++
                ) {
                    // Determine maximum bit width.
                    __m128i pseudoMax128 = _mm_setzero_si128();
                    for(
                            unsigned vecIdx = 0;
                            vecIdx < convert_size<uint64_t, __m128i>(
                                    t_BlockSize64
                            );
                            vecIdx++
                    )
                        pseudoMax128 = _mm_or_si128(
                                pseudoMax128,
                                _mm_load_si128(in128 + vecIdx)
                        );
                    // @todo Why does the first alternative not work?
    #if 0
                    const uint64_t pseudoMax64 = 1
                            | _mm_extract_epi64(pseudoMax128, 0)
                            | _mm_extract_epi64(pseudoMax128, 1);
    #else
                    uint64_t tmp[2];
                    _mm_store_si128(
                            reinterpret_cast<__m128i *>(&tmp),
                            pseudoMax128
                    );
                    const uint64_t pseudoMax64 = 1 | tmp[0] | tmp[1];
    #endif
                    const unsigned bw = effective_bitwidth(pseudoMax64);

                    // Store the bit width to the meta data.
                    outMeta8[blockIdx] = static_cast<uint8_t>(bw);

                    // Pack the data with that bit width.
                    pack_switch(
                            bw,
                            in128,
                            convert_size<uint64_t, __m128i>(t_BlockSize64),
                            out128
                    );
                }
            }

            outCol->set_meta_data(
                    inCount64,
                    convert_size<__m128i, uint8_t>(out128 - initOut128)
            );
            
            return outCol;
        }
    };
    
    template<size_t t_BlockSize64, size_t t_PageSizeBlocks>
    struct morph_t<
            processing_style_t::vec128,
            uncompr_f,
            dynamic_vbp_f<t_BlockSize64, t_PageSizeBlocks>
    > {
        using out_f = uncompr_f;
        using in_f = dynamic_vbp_f<t_BlockSize64, t_PageSizeBlocks>;
        
        static
        const column<out_f> *
        apply(const column<in_f> * inCol) {
            in_f::template check_compatibility<__m128i>();

            const size_t inCount64 = inCol->get_count_values();
            const size_t inCount128 = convert_size<uint64_t, __m128i>(
                    inCount64
            );

            const __m128i * in128 = inCol->get_data();
            const __m128i * const endIn128 = in128 + inCount128;

            auto outCol = new column<out_f>(
                    out_f::get_size_max_byte(inCount64)
            );
            __m128i * out128 = outCol->get_data();

            // Iterate over all pages in the input.
            while(in128 < endIn128) {
                const uint8_t * const inMeta8 = reinterpret_cast<const uint8_t *>(
                    in128
                );
                in128 += convert_size<uint8_t, __m128i>(in_f::m_MetaSize8);
                // Iterate over all blocks in the current input page.
                for(
                        unsigned blockIdx = 0;
                        blockIdx < t_PageSizeBlocks;
                        blockIdx++
                )
                    unpack_switch(
                            inMeta8[blockIdx],
                            in128,
                            out128,
                            convert_size<uint64_t, __m128i>(t_BlockSize64)
                    );
            }

            outCol->set_meta_data(
                    inCount64,
                    convert_size<uint64_t, uint8_t>(inCount64)
            );
            
            return outCol;
        }
    };
    
}
#endif //MORPHSTORE_CORE_MORPHING_DYNAMIC_VBP_H
