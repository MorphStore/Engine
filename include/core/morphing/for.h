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
 * @file for.h
 * @brief A cascade of frame-of-reference and any other format.
 */

#ifndef MORPHSTORE_CORE_MORPHING_FOR_H
#define MORPHSTORE_CORE_MORPHING_FOR_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_primitives.h>
#include <vector/vector_extension_structs.h>

#include <limits>

#include <cstdint>

/**
 * For testing/debugging purposes: If this macro is defined, then the offsets
 * will not be represented using the format t_inner_f, but will instead be
 * stored as uncompressed 64-bit integers.
 */
#undef FOR_UNCOMPRESSED_OFFSETS

#ifdef FOR_UNCOMPRESSED_OFFSETS
#include <cstring>
#endif

#define FOR_STATIC_ASSERTS_VECTOR_EXTENSION \
    static_assert( \
            t_PageSizeBlocks % vector_element_count::value == 0, \
            "for_f: the number of blocks per page must be a multiple of the " \
            "number of elements per vector" \
    );

namespace morphstore {
    
    // ************************************************************************
    // Format
    // ************************************************************************
    
    /**
     * @brief The format of a cascade of frame-of-reference with some other
     * format.
     * 
     * The uncompressed data is subdivided into blocks of `t_BlockSizeLog` data
     * elements each. Within each block, each data element is replaced by its
     * difference to the reference value, which is the minimum data element in
     * the block. After that, these offsets are represented using the format
     * `t_inner_f`. `t_PageSizeBlocks` reference values are stored together
     * before the same number of compressed blocks.
     * 
     * The template parameters are:
     * - `t_BlockSizeLog` The number of logical data elements per block of the
     * cascade. Should be chosen with the cache size in mind.
     * - `t_PageSizeBlocks` The number of reference values to store together
     * before the same number of compressed blocks. For inner formats requiring
     * a SIMD alignment, this parameter should be chosen as a multiple of the
     * number of data elements per vector register.
     * - `t_inner_f` The format to represent the offsets in. This is meant to
     * be a null suppression format.
     */
    template<size_t t_BlockSizeLog, size_t t_PageSizeBlocks, class t_inner_f>
    struct for_f : public format {
        static_assert(
                t_BlockSizeLog % t_inner_f::m_BlockSize == 0,
                "for_f: the logical block size of the cascade must be a "
                "multiple of the logical block size of the inner format"
        );
        static_assert(
                std::is_base_of<representation, t_inner_f>::value,
                "for_f: the template parameter t_inner_f must be a subclass "
                "of representation"
        );
        
        static const size_t m_PageSizeLog = t_PageSizeBlocks * t_BlockSizeLog;
        
        // Assumes that the provided number is a multiple of m_BlockSize.
        static size_t get_size_max_byte(size_t p_CountValues) {
            const size_t countPages = round_up_div(p_CountValues, m_PageSizeLog);
            const size_t countBlocks = p_CountValues / t_BlockSizeLog;
            // @todo In the following, the hardcoded uint64_t must be replaced
            // if we use another base datatype than uint64_t.
#ifdef FOR_UNCOMPRESSED_OFFSETS
            return countPages * convert_size<uint64_t, uint8_t>(t_PageSizeBlocks) +
                    countBlocks * convert_size<uint64_t, uint8_t>(t_BlockSizeLog);
#else
            return countPages * convert_size<uint64_t, uint8_t>(t_PageSizeBlocks) +
                    countBlocks * t_inner_f::get_size_max_byte(t_BlockSizeLog);
#endif
        }
        
        /**
         * Determines the minimum value in the given uncompressed buffer.
         * 
         * @param p_Buffer The uncompressed buffer. Is assumed to contain
         * `t_BlockSizeLog` data elements.
         * @return The minimum value.
         */
        template<class t_vector_extension>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename t_vector_extension::vector_helper_t::base_t
        determine_reference_value(
                const typename t_vector_extension::base_t * p_Buffer
        ) {
            using t_ve = t_vector_extension;
            IMPORT_VECTOR_BOILER_PLATE(t_ve)
            
            using namespace vectorlib;
            
            vector_t minVec =
                    load<t_ve, iov::ALIGNED, vector_size_bit::value>(p_Buffer);
            for(
                    unsigned baseIdx = vector_element_count::value;
                    baseIdx < t_BlockSizeLog;
                    baseIdx += vector_element_count::value
            )
                minVec = min<t_ve>::apply(
                        minVec,
                        load<t_ve, iov::ALIGNED, vector_size_bit::value>(
                                p_Buffer + baseIdx
                        )
                );

            // @todo Use vectorlib::hmin-primitive when it exists.
            MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value)
            base_t tmp[vector_element_count::value];
            store<t_ve, iov::ALIGNED, vector_size_bit::value>(tmp, minVec);
            base_t minBase = std::numeric_limits<base_t>::max();
            for(unsigned i = 0; i < vector_element_count::value; i++)
                if(tmp[i] < minBase)
                    minBase = tmp[i];

            return minBase;
        }
        
        static const size_t m_BlockSize = t_BlockSizeLog;
    };
    

    // ************************************************************************
    // Morph-operators (batch-level)
    // ************************************************************************
    
    // @todo How to tailor t_inner_f to t_vector_extension?
    
    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            size_t t_BlockSizeLog,
            size_t t_PageSizeBlocks,
            class t_inner_f
    >
    struct morph_batch_t<
            t_vector_extension,
            for_f<t_BlockSizeLog, t_PageSizeBlocks, t_inner_f>,
            uncompr_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
                
        FOR_STATIC_ASSERTS_VECTOR_EXTENSION
                
        using dst_f = for_f<t_BlockSizeLog, t_PageSizeBlocks, t_inner_f>;
        
        static void apply(
                const uint8_t * & p_In8, uint8_t * & p_Out8, size_t p_CountLog
        ) {
            using namespace vectorlib;
            
            const base_t * inBase = reinterpret_cast<const base_t *>(p_In8);
            
            MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value)
            base_t tmp[t_BlockSizeLog];
            
            // Iterate over all input pages (the complete pages and the final
            // incomplete page).
            for(size_t countLogCompr = 0; countLogCompr < p_CountLog;) {
                base_t * const outMeta = reinterpret_cast<base_t *>(p_Out8);
                p_Out8 += convert_size<base_t, uint8_t>(t_PageSizeBlocks);
                // Iterate over all blocks in the current input page.
                for(
                        unsigned blockIdx = 0;
                        blockIdx < t_PageSizeBlocks && countLogCompr < p_CountLog;
                        blockIdx++, countLogCompr += t_BlockSizeLog
                ) {
                    // FOR part.
                    const base_t ref =
                            dst_f::template determine_reference_value<t_ve>(
                                inBase
                            );
                    outMeta[blockIdx] = ref;
                    const vector_t refVec = set1<
                            t_ve, vector_base_t_granularity::value
                    >(ref);
                    for(
                            size_t i = 0;
                            i < t_BlockSizeLog;
                            i += vector_element_count::value
                    ) {
                        store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                                tmp + i,
                                sub<t_ve>::apply(
                                        load<
                                                t_ve,
                                                iov::ALIGNED,
                                                vector_size_bit::value
                                        >(inBase + i),
                                        refVec
                                )
                        );
                    }
                    inBase += t_BlockSizeLog;

                    // Inner part.
#ifdef FOR_UNCOMPRESSED_OFFSETS
                    const size_t tmpSizeByte = t_BlockSizeLog * sizeof(base_t);
                    memcpy(p_Out8, tmp, tmpSizeByte);
                    p_Out8 += tmpSizeByte;
#else
                    const uint8_t * tmp8 =
                            reinterpret_cast<const uint8_t *>(tmp);
                    morph_batch<t_ve, t_inner_f, uncompr_f>(
                            tmp8, p_Out8, t_BlockSizeLog
                    );
#endif
                }
            }
            
            p_In8 = reinterpret_cast<const uint8_t *>(inBase);
        }
    };
    
    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            size_t t_BlockSizeLog,
            size_t t_PageSizeBlocks,
            class t_inner_f
    >
    struct morph_batch_t<
            t_vector_extension,
            uncompr_f,
            for_f<t_BlockSizeLog, t_PageSizeBlocks, t_inner_f>
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
                
        FOR_STATIC_ASSERTS_VECTOR_EXTENSION
        
        using src_f = for_f<t_BlockSizeLog, t_PageSizeBlocks, t_inner_f>;
        
        static void apply(
                const uint8_t * & p_In8, uint8_t * & p_Out8, size_t p_CountLog
        ) {
            using namespace vectorlib;
            
            MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value)
            base_t tmp[t_BlockSizeLog];
            
            base_t * outBase = reinterpret_cast<base_t *>(p_Out8);
            
            // Iterate over all complete pages and possibly the final
            // incomplete page in the input.
            for(size_t countLogDecompr = 0; countLogDecompr < p_CountLog;) {
                const base_t * const inMetaBase =
                        reinterpret_cast<const base_t *>(p_In8);
                p_In8 += convert_size<base_t, uint8_t>(t_PageSizeBlocks);
                for(
                        unsigned blockIdx = 0;
                        blockIdx < t_PageSizeBlocks && countLogDecompr < p_CountLog;
                        blockIdx++, countLogDecompr += t_BlockSizeLog
                ) {
                    // Inner part.
#ifdef FOR_UNCOMPRESSED_OFFSETS
                    const size_t tmpSizeByte = t_BlockSizeLog * sizeof(base_t);
                    memcpy(tmp, p_In8, tmpSizeByte);
                    p_In8 += tmpSizeByte;
#else
                    uint8_t * tmp8 = reinterpret_cast<uint8_t *>(tmp);
                    morph_batch<t_ve, uncompr_f, t_inner_f>(
                            p_In8, tmp8, t_BlockSizeLog
                    );
#endif
                    
                    // FOR part.
                    const vector_t refVec = set1<
                            t_ve, vector_base_t_granularity::value
                    >(inMetaBase[blockIdx]);
                    for(
                            size_t i = 0;
                            i < t_BlockSizeLog;
                            i += vector_element_count::value
                    ) {
                        store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                                outBase + i,
                                add<t_ve>::apply(
                                        load<
                                                t_ve,
                                                iov::ALIGNED,
                                                vector_size_bit::value
                                        >(tmp + i),
                                        refVec
                                )
                        );
                    }
                    outBase += t_BlockSizeLog;
                }
            }
            
            p_Out8 = reinterpret_cast<uint8_t *>(outBase);
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
            class t_inner_f,
            template<
                    class /*t_vector_extension*/, class ... /*t_extra_args*/
            > class t_op_vector,
            class ... t_extra_args
    >
    struct decompress_and_process_batch<
            t_vector_extension,
            for_f<
                    t_BlockSizeLog,
                    t_PageSizeBlocks,
                    t_inner_f
            >,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
                
        FOR_STATIC_ASSERTS_VECTOR_EXTENSION
        
        using in_f = for_f<t_BlockSizeLog, t_PageSizeBlocks, t_inner_f>;
        
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountInLog,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & p_State
        ) {
            using namespace vectorlib;
            
            MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value)
            base_t tmp[t_BlockSizeLog];
            
            // Iterate over all complete pages and possibly the final
            // incomplete page in the input.
            for(size_t countLogDecompr = 0; countLogDecompr < p_CountInLog;) {
                const base_t * const inMetaBase =
                        reinterpret_cast<const base_t *>(p_In8);
                p_In8 += convert_size<base_t, uint8_t>(t_PageSizeBlocks);
                for(
                        unsigned blockIdx = 0;
                        blockIdx < t_PageSizeBlocks && countLogDecompr < p_CountInLog;
                        blockIdx++, countLogDecompr += t_BlockSizeLog
                ) {
                    // Inner part.
#ifdef FOR_UNCOMPRESSED_OFFSETS
                    const size_t tmpSizeByte = t_BlockSizeLog * sizeof(base_t);
                    memcpy(tmp, p_In8, tmpSizeByte);
                    p_In8 += tmpSizeByte;
#else
                    uint8_t * tmp8 = reinterpret_cast<uint8_t *>(tmp);
                    morph_batch<t_ve, uncompr_f, t_inner_f>(
                            p_In8, tmp8, t_BlockSizeLog
                    );
#endif
                    
                    // FOR part.
                    const vector_t refVec = set1<
                            t_ve, vector_base_t_granularity::value
                    >(inMetaBase[blockIdx]);
                    for(
                            size_t i = 0;
                            i < t_BlockSizeLog;
                            i += vector_element_count::value
                    ) {
                        t_op_vector<t_ve, t_extra_args ...>::apply(
                                add<t_ve>::apply(
                                        load<
                                                t_ve,
                                                iov::ALIGNED,
                                                vector_size_bit::value
                                        >(tmp + i),
                                        refVec
                                ),
                                p_State
                        );
                    }
                }
            }
        }
    };
    
}
#endif //MORPHSTORE_CORE_MORPHING_FOR_H
