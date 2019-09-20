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
 * @file delta.h
 * @brief A cascade of delta coding and any other format.
 */

#ifndef MORPHSTORE_CORE_MORPHING_DELTA_H
#define MORPHSTORE_CORE_MORPHING_DELTA_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/utils/basic_types.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_primitives.h>
#include <vector/vector_extension_structs.h>

#include <cstdint>

/**
 * For testing/debugging purposes: If this macro is defined, then the deltas
 * will not be represented using the format t_inner_f, but will instead be
 * stored as uncompressed 64-bit integers.
 */
#undef DELTA_UNCOMPRESSED_DELTAS

#ifdef DELTA_UNCOMPRESSED_DELTAS
#include <cstring>
#endif

namespace morphstore {
    
    // ************************************************************************
    // Format
    // ************************************************************************
    
    // @todo Currently, each block of the cascade begins with a
    // non-delta-compressed vector, which is unsuitable for the inner
    // compression.
    /**
     * @brief The format of a cascade of delta coding with some other format.
     * 
     * The uncompressed data is subdivided into blocks of `t_BlockSizeLog` data
     * elements each. Within each block, each data element is replaced by its
     * difference to its `t_Step`-th predecessor, except for the first `t_Step`
     * data elements in the block. After that, these deltas are represented
     * using the format `t_inner_f`.
     * 
     * Note that this format can be used for both sorted and unsorted data, but
     * for the latter, it will usually be quite unsuitable.
     * 
     * The template parameters are:
     * - `t_BlockSizeLog` The number of logical data elements per block of the
     * cascade. Should be chosen with the cache size in mind.
     * - `t_Step` The step width of the difference calculation. This is meant
     * to be the number of data elements per vector of the chosen vector
     * extension.
     * - `t_inner_f` The format to represent the deltas in. This is meant to be
     * a null suppression format.
     */
    template<size_t t_BlockSizeLog, unsigned t_Step, class t_inner_f>
    struct delta_f : public format {
        static_assert(
                t_BlockSizeLog % t_inner_f::m_BlockSize == 0,
                "delta_f: the logical block size of the cascade must be a "
                "multiple of the logical block size of the inner format"
        );
        static_assert(
                std::is_base_of<representation, t_inner_f>::value,
                "delta: the template parameter t_inner_f must be a subclass "
                "of representation"
        );
        
        // Assumes that the provided number is a multiple of m_BlockSize.
        static size_t get_size_max_byte(size_t p_CountValues) {
            // DELTA has no meta data to account for. In the worst case, the
            // deltas are 64-bit numbers which is the general worst-case for
            // null suppression algorithms anyway, so the pessimistic size
            // estimation can be delegated to the inner format.
#ifdef DELTA_UNCOMPRESSED_DELTAS
            return convert_size<uint64_t, uint8_t>(p_CountValues);
#else
            return t_inner_f::get_size_max_byte(p_CountValues);
#endif
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
    
    template<class t_vector_extension, size_t t_BlockSizeLog, class t_inner_f>
    struct morph_batch_t<
            t_vector_extension,
            delta_f<
                    t_BlockSizeLog,
                    t_vector_extension::vector_helper_t::element_count::value,
                    t_inner_f
            >,
            uncompr_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using dst_f = delta_f<
                t_BlockSizeLog, vector_element_count::value, t_inner_f
        >;
        
        static void apply(
                const uint8_t * & p_In8, uint8_t * & p_Out8, size_t p_CountLog
        ) {
            using namespace vectorlib;
            
            MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value)
            base_t tmp[t_BlockSizeLog];

            const base_t * inBase = reinterpret_cast<const base_t *>(p_In8);
            
            const size_t countBlocks = p_CountLog / t_BlockSizeLog;
            for(size_t blockIdx = 0; blockIdx < countBlocks; blockIdx++) {
                // Delta part.
                vector_t lastVec = load<
                        t_ve, iov::ALIGNED, vector_size_bit::value
                >(inBase);
                inBase += vector_element_count::value;
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(tmp, lastVec);
                for(
                        size_t valIdxInBlock = vector_element_count::value;
                        valIdxInBlock < t_BlockSizeLog;
                        valIdxInBlock += vector_element_count::value
                ) {
                    const vector_t curVec = load<
                            t_ve, iov::ALIGNED, vector_size_bit::value
                    >(inBase);
                    inBase += vector_element_count::value;
                    store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                            tmp + valIdxInBlock,
                            sub<t_ve>::apply(curVec, lastVec)
                    );
                    lastVec = curVec;
                }
                
                // Inner part.
#ifdef DELTA_UNCOMPRESSED_DELTAS
                const size_t tmpSizeByte = t_BlockSizeLog * sizeof(base_t);
                memcpy(p_Out8, tmp, tmpSizeByte);
                p_Out8 += tmpSizeByte;
#else
                const uint8_t * tmp8 = reinterpret_cast<const uint8_t *>(tmp);
                morph_batch<t_ve, t_inner_f, uncompr_f>(
                        tmp8, p_Out8, t_BlockSizeLog
                );
#endif
            }
            
            p_In8 = reinterpret_cast<const uint8_t *>(inBase);
        }
    };
    
    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    
    template<class t_vector_extension, size_t t_BlockSizeLog, class t_inner_f>
    struct morph_batch_t<
            t_vector_extension,
            uncompr_f,
            delta_f<
                    t_BlockSizeLog,
                    t_vector_extension::vector_helper_t::element_count::value,
                    t_inner_f
            >
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using src_f = delta_f<
                t_BlockSizeLog, vector_element_count::value, t_inner_f
        >;
        
        static void apply(
                const uint8_t * & p_In8, uint8_t * & p_Out8, size_t p_CountLog
        ) {
            using namespace vectorlib;
            
            MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value)
            base_t tmp[t_BlockSizeLog];
            
            base_t * outBase = reinterpret_cast<base_t *>(p_Out8);
            
            const size_t countBlocks = p_CountLog / t_BlockSizeLog;
            for(size_t blockIdx = 0; blockIdx < countBlocks; blockIdx++) {
                // Inner part.
#ifdef DELTA_UNCOMPRESSED_DELTAS
                const size_t tmpSizeByte = t_BlockSizeLog * sizeof(base_t);
                memcpy(tmp, p_In8, tmpSizeByte);
                p_In8 += tmpSizeByte;
#else
                uint8_t * tmp8 = reinterpret_cast<uint8_t *>(tmp);
                morph_batch<t_ve, uncompr_f, t_inner_f>(
                        p_In8, tmp8, t_BlockSizeLog
                );
#endif
                
                // Delta part.
                vector_t prefixSumVec = load<
                        t_ve, iov::ALIGNED, vector_size_bit::value
                >(tmp);
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        outBase, prefixSumVec
                );
                outBase += vector_element_count::value;
                for(
                        size_t valIdxInBlock = vector_element_count::value;
                        valIdxInBlock < t_BlockSizeLog;
                        valIdxInBlock += vector_element_count::value
                ) {
                    prefixSumVec = add<t_ve>::apply(
                            prefixSumVec,
                            load<t_ve, iov::ALIGNED, vector_size_bit::value>(
                                    tmp + valIdxInBlock
                            )
                    );
                    store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                            outBase, prefixSumVec
                    );
                    outBase += vector_element_count::value;
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
            class t_inner_f,
            template<
                    class /*t_vector_extension*/, class ... /*t_extra_args*/
            > class t_op_vector,
            class ... t_extra_args
    >
    struct decompress_and_process_batch<
            t_vector_extension,
            delta_f<
                    t_BlockSizeLog,
                    t_vector_extension::vector_helper_t::element_count::value,
                    t_inner_f
            >,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using in_f = delta_f<
                t_BlockSizeLog, vector_element_count::value, t_inner_f
        >;
        
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountInLog,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & p_State
        ) {
            using namespace vectorlib;
            
            MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value)
            base_t tmp[t_BlockSizeLog];
            
            const size_t countBlocks = p_CountInLog / t_BlockSizeLog;
            for(size_t blockIdx = 0; blockIdx < countBlocks; blockIdx++) {
                // Inner part.
#ifdef DELTA_UNCOMPRESSED_DELTAS
                const size_t tmpSizeByte = t_BlockSizeLog * sizeof(base_t);
                memcpy(tmp, p_In8, tmpSizeByte);
                p_In8 += tmpSizeByte;
                
#else
                uint8_t * tmp8 = reinterpret_cast<uint8_t *>(tmp);
                morph_batch<t_ve, uncompr_f, t_inner_f>(
                        p_In8, tmp8, t_BlockSizeLog
                );
#endif
                
                // Delta part.
                vector_t prefixSumVec = load<
                        t_ve, iov::ALIGNED, vector_size_bit::value
                >(tmp);
                t_op_vector<t_ve, t_extra_args ...>::apply(
                        prefixSumVec, p_State
                );
                for(
                        size_t valIdxInBlock = vector_element_count::value;
                        valIdxInBlock < t_BlockSizeLog;
                        valIdxInBlock += vector_element_count::value
                ) {
                    prefixSumVec = add<t_ve>::apply(
                            prefixSumVec,
                            load<t_ve, iov::ALIGNED, vector_size_bit::value>(
                                    tmp + valIdxInBlock
                            )
                    );
                    t_op_vector<t_ve, t_extra_args ...>::apply(
                            prefixSumVec, p_State
                    );
                }
            }
        }
    };
    
}
#endif //MORPHSTORE_CORE_MORPHING_DELTA_H
