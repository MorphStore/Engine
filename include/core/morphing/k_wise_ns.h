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
 * @file k_wise_ns.h
 * @brief The compression format/algorithm k-Wise Null Suppression.
 * 
 * This is a 64-bit implementation of k-Wise Null Suppression originally
 * proposed in the following paper:
 * 
 * Benjamin Schlegel, Rainer Gemulla, Wolfgang Lehner: Fast integer compression
 * using SIMD instructions. DaMoN 2010: 34-40
 * 
 * Currently, there is only an implementation for SSE, i.e., 2-Wise Null
 * Suppression on 64-bit integers.
 * 
 * @todo Documentation of the memory layout.
 * @todo Group multiple descriptors together.
 */

#ifndef MORPHSTORE_CORE_MORPHING_K_WISE_NS_H
#define MORPHSTORE_CORE_MORPHING_K_WISE_NS_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_primitives.h>
#include <vector/vector_extension_structs.h>

#include <cstdint>

namespace morphstore {
    
    // ************************************************************************
    // Format
    // ************************************************************************
    
    template<size_t t_BlockSizeLog>
    class k_wise_ns_f : public format {
        
        static const size_t m_ShuffleMaskCount =
            1 << (effective_bitwidth(sizeof(uint64_t) - 1) * t_BlockSizeLog);
        
        static const size_t m_TableSizeByte =
            m_ShuffleMaskCount * t_BlockSizeLog * sizeof(uint64_t);
        
        // Having a separate function for the creation of each of the three
        // tables is somewhat redundant. However, this way it is easier to have
        // the tables initialized without having to call some function at
        // runtime (which could easily be forgotten).
        // Note that performance does not matter for the initialization of the
        // tables since it is done only once.
        
        static const uint8_t * build_table_shuffle_mask_compr() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
            uint8_t * res = create_aligned_ptr(reinterpret_cast<uint8_t *>(
                    malloc(get_size_with_alignment_padding(m_TableSizeByte))
            ));
#else
            uint8_t * res = reinterpret_cast<uint8_t *>(
                    malloc(m_TableSizeByte)
            );
#endif
            
            for(uint8_t descr = 0; descr < m_ShuffleMaskCount; descr++) {
                const size_t vecOffset = (m_ShuffleMaskCount - descr - 1) *
                        t_BlockSizeLog * sizeof(uint64_t);
                unsigned f = 0;

                unsigned zeroBytes[2];
                zeroBytes[0] =  descr & 0b000111;
                zeroBytes[1] = (descr & 0b111000) >> 3;

                for(size_t elInVec = 0; elInVec < t_BlockSizeLog; elInVec++) {
                    for(size_t byteInEl = 0; byteInEl < sizeof(uint64_t); byteInEl++) {
                        unsigned byteInVec = elInVec * sizeof(uint64_t) + byteInEl;
                        if(byteInEl <= zeroBytes[elInVec])
                            res[vecOffset + f++] = byteInVec;
                    }
                }
            }
            
            return res;
        }
        
        static const uint8_t * build_table_shuffle_mask_decompr() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
            uint8_t * res = create_aligned_ptr(reinterpret_cast<uint8_t *>(
                    malloc(get_size_with_alignment_padding(m_TableSizeByte))
            ));
#else
            uint8_t * res = reinterpret_cast<uint8_t *>(
                    malloc(m_TableSizeByte)
            );
#endif
            
            for(uint8_t descr = 0; descr < m_ShuffleMaskCount; descr++) {
                const size_t vecOffset = (m_ShuffleMaskCount - descr - 1) *
                        t_BlockSizeLog * sizeof(uint64_t);
                unsigned f = 0;

                unsigned zeroBytes[2];
                zeroBytes[0] =  descr & 0b000111;
                zeroBytes[1] = (descr & 0b111000) >> 3;

                for(size_t elInVec = 0; elInVec < t_BlockSizeLog; elInVec++) {
                    for(size_t byteInEl = 0; byteInEl < sizeof(uint64_t); byteInEl++) {
                        unsigned byteInVec = elInVec * sizeof(uint64_t) + byteInEl;
                        if(byteInEl <= zeroBytes[elInVec])
                            res[vecOffset + byteInVec] = f++;
                        else
                            res[vecOffset + byteInVec] = 0x80;
                    }
                }
            }
            
            return res;
        }
        
        static const size_t * build_table_block_size_byte() {
            size_t * res = reinterpret_cast<size_t *>(
                    malloc(m_ShuffleMaskCount * sizeof(size_t))
            );
            
            for(uint8_t descr = 0; descr < m_ShuffleMaskCount; descr++) {
                unsigned zeroBytes[2];
                zeroBytes[0] =  descr & 0b000111;
                zeroBytes[1] = (descr & 0b111000) >> 3;

                res[descr] = t_BlockSizeLog * sizeof(uint64_t);
                for(size_t elInVec = 0; elInVec < t_BlockSizeLog; elInVec++)
                    res[descr] -= zeroBytes[elInVec];
            }
            
            return res;
        }
        
    public:
        // Assumes that the provided number is a multiple of m_BlockSize.
        static size_t get_size_max_byte(size_t p_CountValues) {
            // In the worst case, no data element can be compressed, plus there
            // is a one-byte descriptor for each vector.
            return convert_size<uint64_t, uint8_t>(p_CountValues) + 
                    p_CountValues / m_BlockSize;
        }
        
        static const size_t m_BlockSize = t_BlockSizeLog;
        
        static const uint8_t * const m_ShuffleMaskCompr;
        
        static const uint8_t * const m_ShuffleMaskDecompr;
        
        static const size_t * const m_BlockSizeByte;
    };
    
    template<size_t t_BlockSizeLog>
    const uint8_t * const k_wise_ns_f<t_BlockSizeLog>::m_ShuffleMaskCompr =
            build_table_shuffle_mask_compr();
    
    template<size_t t_BlockSizeLog>
    const uint8_t * const k_wise_ns_f<t_BlockSizeLog>::m_ShuffleMaskDecompr =
            build_table_shuffle_mask_decompr();
    
    template<size_t t_BlockSizeLog>
    const size_t * const k_wise_ns_f<t_BlockSizeLog>::m_BlockSizeByte =
            build_table_block_size_byte();
    
    
    // ************************************************************************
    // Morph-operators (batch-level)
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    
#ifdef SSE
    template<>
    struct morph_batch_t<
            vectorlib::sse<vectorlib::v128<uint64_t> >,
            k_wise_ns_f<
                    vectorlib::sse<
                            vectorlib::v128<uint64_t>
                    >::vector_helper_t::element_count::value
            >,
            uncompr_f
    > {
        using t_ve = vectorlib::sse<vectorlib::v128<uint64_t> >;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using dst_f = k_wise_ns_f<vector_element_count::value>;
        
        static void apply(
                const uint8_t * & p_In8, uint8_t * & p_Out8, size_t p_CountLog
        ) {
            using namespace vectorlib;
            
            const base_t * inBase = reinterpret_cast<const base_t *>(p_In8);
            const vector_t * const masks = reinterpret_cast<const vector_t *>(
                    dst_f::m_ShuffleMaskCompr
            );
            
            for(
                    size_t i = 0;
                    i < p_CountLog;
                    i += vector_element_count::value
            ) {
                const unsigned zeroBytes0 = zero_bytes(inBase[i    ]);
                const unsigned zeroBytes1 = zero_bytes(inBase[i + 1]);
                uint8_t descr = (zeroBytes1 << 3) | zeroBytes0;
                *p_Out8++ = descr;

                store<t_ve, iov::UNALIGNED, vector_size_bit::value>(
                        reinterpret_cast<base_t *>(p_Out8),
                        _mm_shuffle_epi8(
                                load<
                                        t_ve,
                                        iov::ALIGNED,
                                        vector_size_bit::value
                                >(inBase + i),
                                load<
                                        t_ve,
                                        iov::ALIGNED,
                                        vector_size_bit::value
                                >(
                                        reinterpret_cast<const base_t *>(
                                                masks + descr
                                        )
                                )
                        )
                );
                p_Out8 += dst_f::m_BlockSizeByte[descr];
            }
            
            p_In8 += convert_size<uint64_t, uint8_t>(p_CountLog);
        }
    };
#endif
    
    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    
#ifdef SSE
    template<>
    struct morph_batch_t<
            vectorlib::sse<vectorlib::v128<uint64_t> >,
            uncompr_f,
            k_wise_ns_f<
                    vectorlib::sse<
                            vectorlib::v128<uint64_t>
                    >::vector_helper_t::element_count::value
            >
    > {
        using t_ve = vectorlib::sse<vectorlib::v128<uint64_t> >;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using src_f = k_wise_ns_f<vector_element_count::value>;
        
        static void apply(
                const uint8_t * & p_In8, uint8_t * & p_Out8, size_t p_CountLog
        ) {
            using namespace vectorlib;
            
            base_t * outBase = reinterpret_cast<base_t *>(p_Out8);
            const vector_t * const masks = reinterpret_cast<const vector_t *>(
                    src_f::m_ShuffleMaskDecompr
            );
            
            for(size_t i = 0; i < p_CountLog; i += vector_element_count::value) {
                const uint8_t descr = *p_In8++;
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        outBase,
                        _mm_shuffle_epi8(
                                load<
                                        t_ve,
                                        iov::UNALIGNED,
                                        vector_size_bit::value
                                >(reinterpret_cast<const base_t *>(p_In8)),
                                load<
                                        t_ve,
                                        iov::ALIGNED,
                                        vector_size_bit::value
                                >(reinterpret_cast<const base_t *>(
                                        masks + descr
                                ))
                        )
                );
                p_In8 += src_f::m_BlockSizeByte[descr];
                outBase += vector_element_count::value;
            }
            
            p_Out8 += convert_size<uint64_t, uint8_t>(p_CountLog);
        }
    };
#endif
    
    // ************************************************************************
    // Interfaces for accessing compressed data
    // ************************************************************************

    // ------------------------------------------------------------------------
    // Sequential read
    // ------------------------------------------------------------------------
    
#ifdef SSE
    template<
            template<
                    class /*t_vector_extension*/, class ... /*t_extra_args*/
            > class t_op_vector,
            class ... t_extra_args
    >
    struct decompress_and_process_batch<
            vectorlib::sse<vectorlib::v128<uint64_t>>,
            k_wise_ns_f<
                    vectorlib::sse<
                            vectorlib::v128<uint64_t>
                    >::vector_helper_t::element_count::value
            >,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = vectorlib::sse<vectorlib::v128<uint64_t> >;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using in_f = k_wise_ns_f<vector_element_count::value>;
        
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountInLog,
                typename t_op_vector<
                        t_ve,
                        t_extra_args ...
                >::state_t & p_State
        ) {
            using namespace vectorlib;
            
            const vector_t * const masks = reinterpret_cast<const vector_t *>(
                    in_f::m_ShuffleMaskDecompr
            );
            
            for(size_t i = 0; i < p_CountInLog; i += vector_element_count::value) {
                const uint8_t descr = *p_In8++;
                t_op_vector<t_ve, t_extra_args ...>::apply(
                        _mm_shuffle_epi8(
                                load<
                                        t_ve,
                                        iov::UNALIGNED,
                                        vector_size_bit::value
                                >(reinterpret_cast<const base_t *>(p_In8)),
                                load<
                                        t_ve,
                                        iov::ALIGNED,
                                        vector_size_bit::value
                                >(reinterpret_cast<const base_t *>(
                                        masks + descr
                                ))
                        ),
                        p_State
                );
                p_In8 += in_f::m_BlockSizeByte[descr];
            }
        }
    };
#endif
    
}
#endif //MORPHSTORE_CORE_MORPHING_K_WISE_NS_H
