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
 * @file write_iterator.h
 * @brief Interfaces and default implementations for writing data to a column
 * in any format.
 */

#ifndef MORPHSTORE_CORE_MORPHING_WRITE_ITERATOR_H
#define MORPHSTORE_CORE_MORPHING_WRITE_ITERATOR_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/morph.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <tuple>

#include <cstdint>
#include <cstring>

namespace morphstore {
    
    // @todo If a write-iterator knows that its data is sorted then it could do
    // more optimizations, e.g., when determining the bit width, only the last
    // value in the block would need to be considered, since it is always the
    // greatest.
    
    /**
     * @brief The interface for writing compressed data selectively.
     * 
     * The default implementation buffers the appended data elements in
     * uncompressed form first. When the internal buffer is full, then the
     * batch-level morph-operator is used to compress the buffer and append the
     * compresed data to the output column's data buffer. **This interface does
     * not need to be implemented for new formats**, since the default
     * implementation can always be used as long as there is a specialization
     * of the batch-level morph-operator for the respective output format.
     */
    template<class t_vector_extension, class t_format>
    class selective_write_iterator {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        uint8_t * m_Out;
        const uint8_t * const m_InitOut;
        // The largest multiple of the format's block size which is not smaller
        // than 2048 logical data elements respectively 16ki bytes of
        // uncompressed data.
        // @todo Think about the buffer size.
        static const size_t m_CountBuffer = round_up_to_multiple(
                t_format::m_BlockSize, 2048
        );
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
            morph_batch<t_ve, t_format, uncompr_f>(
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
        
        /**
         * @brief Stores the elements of the given data vector selected by the
         * given mask to the output.
         * 
         * Internally, buffering may take place, such that it is not guaranteed
         * that the data is stored to the output immediately.
         * 
         * `done` must be called after the last call to this function to
         * guarantee that the data is stored to the output in any case.
         * 
         * @param p_Data
         * @param p_Mask
         */
        MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(
                vector_t p_Data, vector_mask_t p_Mask
        ) {
            write(
                    p_Data,
                    p_Mask,
                    vectorlib::count_matches<t_ve>::apply(p_Mask)
            );
        }
        

        /**
         * @brief Makes sure that all possibly buffered data is stored to the
         * output and returns useful information for further processing.
         * 
         * This function should always be called after the last call to
         * `write`.
         * 
         * For compressed output formats, the output's uncompressed rest part
         * is initialized if the number of data elements stored using this
         * instance is not compressible in the output format.
         * 
         * @return A tuple with the following elements:
         * 1. The size of the output's *compressed* part in bytes.
         * 2. `true` if the output's *uncompressed* part has been initialized,
         *    `false` otherwise.
         * 3. A pointer to the end of the stored (un)compressed data, which can
         *    be used to continue storing data to the output.
         */
        std::tuple<size_t, bool, uint8_t *> done() {
            const size_t countLog = m_Buffer - m_StartBuffer;
            bool startedUncomprPart = false;
            size_t outSizeComprByte;
            if(countLog) {
                const size_t outCountLogCompr = round_down_to_multiple(
                        countLog, t_format::m_BlockSize
                );

                const uint8_t * buffer8 = reinterpret_cast<uint8_t *>(
                        m_StartBuffer
                );
                morph_batch<t_ve, t_format, uncompr_f>(
                    buffer8, m_Out, outCountLogCompr
                );
                outSizeComprByte = m_Out - m_InitOut;

                const size_t outCountLogRest = countLog - outCountLogCompr;
                if(outCountLogRest) {
                    m_Out = create_aligned_ptr(m_Out);
                    const size_t sizeOutLogRest =
                            uncompr_f::get_size_max_byte(outCountLogRest);
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

        /**
         * @brief Returns the number of logical data elements that were stored
         * using this instance.
         * @return The number of logical data elements stored using this
         * instance.
         */
        size_t get_count_values () const {
            return m_Count;
        }
    };
    
    /**
     * @brief The interface for writing compressed data non-selectively.
     *
     * Currently, we have no implementations for non-selective write-iterators
     * yet. Therefore, we delegate to the selective counterpart. This enables
     * the implementation of non-selective operators against the non-selective
     * interface.
     */
    // @todo Implement the non-selective write-iterators.
    template<class t_vector_extension, class t_format>
    class nonselective_write_iterator {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)

        selective_write_iterator<t_vector_extension, t_format> m_Wit;

    public:
        nonselective_write_iterator(uint8_t * p_Out) : m_Wit(p_Out) {
            //
        };

        /**
         * @brief Stores the given data vector to the output.
         * 
         * Internally, buffering may take place, such that it is not guaranteed
         * that the data is stored to the output immediately.
         * 
         * `done` must be called after the last call to this function to
         * guarantee that the data is stores to the output in any case.
         * 
         * @param p_Data
         */
        MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(vector_t p_Data) {
            m_Wit.write(
                    p_Data,
                    bitwidth_max<vector_mask_t>(vector_element_count::value)
            );
        }

        /**
         * @brief Makes sure that all possibly buffered data is stored to the
         * output and returns useful information for further processing.
         * 
         * This function should always be called after the last call to
         * `write`.
         * 
         * For compressed output formats, the output's uncompressed rest part
         * is initialized if the number of data elements stored using this
         * instance is not compressible in the output format.
         * 
         * @return A tuple with the following elements:
         * 1. The size of the output's *compressed* part in bytes.
         * 2. `true` if the output's *uncompressed* part has been initialized,
         *    `false` otherwise.
         * 3. A pointer to the end of the stored (un)compressed data, which can
         *    be used to continue storing data to the output.
         */
        std::tuple<size_t, bool, uint8_t *> done() {
            return m_Wit.done();
        }
    };

}
#endif //MORPHSTORE_CORE_MORPHING_WRITE_ITERATOR_H
