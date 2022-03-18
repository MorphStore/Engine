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
 * @file write_iterator_IR.h
 * @brief Interfaces and default implementations for
 *              (1) handling IR-transformations
 *                                  +
 *              (2) writing data to a column in any format.
 *
 *        General remark: The query operators are implemented against the interfaces
 *        `selective_write_iterator` and `nonselective_write_iterator_IR`. Both must have
 *        the member functions `done()` and `get_count_values`. Furthermore, both
 *        must provide a `write` member function, however, the parameters differ.
 *        The selective one must provide `write` with (i) a data vector and a mask,
 *        and (ii) with a data vector, a mask, and the popcount of the mask. The
 *        non-selective interface must provide `write` with only a data vector.
 *
 *        This is the complete interface that must be implemented for each template
 *        specialization for some format.
 *
 *        All other member functions etc. in this file (including the base class
 *        `write_iterator_base_IR` are specific to the buffered default implementation,
 *        which works for all formats.
 *
 *        Generally default implementation of write_iterator_base_IR uses 3 buffers internally:
 *          (1) IR-Collection-Buffer:     collects results from vector-register layer (t_op<>) +
 *                                        executes IR-transformation to IR-Transformation-Buffer, if capacity reached
 *          (2) IR-Transformation-Buffer: contains uncompressed IR-transformed data +
 *                                        copies elements to Morph-Buffer, executes compress_buffer() of Morph-Buffer
 *          (3) Morph-Buffer:             compresses data to output-column in any format
 *
 */

#ifndef MORPHSTORE_CORE_MORPHING_WRITE_ITERATOR_IR_H
#define MORPHSTORE_CORE_MORPHING_WRITE_ITERATOR_IR_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/morph.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <core/morphing/intermediates/position_list.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/morphing/intermediates/representation.h>
#include <core/morphing/intermediates/transformations/transformation_algorithms.h>

#include <tuple>

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace morphstore {

    /**
     * @brief General default implementation of write_iterator_base_IR.
     *        Uses 3 buffers internally.
     *
     */
    template<
            class t_vector_extension,
            class t_format,
            class t_IR_dst_f = uncompr_f, // By default, we set them to uncompr_f, so that we can leave it out (blank),
            class t_IR_src_f = uncompr_f, //  + IR transformations are always done on uncompressed data
            typename std::enable_if_t<
                    // enable only if (both formats == uncompr_f) OR (both are IR-types)
                    (std::is_same<t_IR_src_f, uncompr_f>::value && std::is_same<t_IR_dst_f, uncompr_f>::value) || // to support existing implementations
                    (is_intermediate_representation_t<t_IR_src_f>::value && is_intermediate_representation_t<t_IR_dst_f>::value)
            , int> = 0
    >
    class write_iterator_base_IR {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

        // Morph-Buffer output-pointer
        uint8_t * m_Morph_Out;
        const uint8_t * const m_Morph_InitOut;
    public:
        // Morph-Buffer total count
        static const size_t m_Morph_CountBuffer = round_up_to_multiple(
                t_format::m_BlockSize, 2048
        );

        // IR-Transformation-Buffer total count
        // In general, buffer can hold up to 2048 uncompressed data elements (internal Lx-cache-resident buffer of 16ki bytes)
        // For uncompr_, we assume position-list, i.e. 2048
        static const size_t m_IR_Trans_CountBuffer =
                (std::is_same<t_IR_dst_f, uncompr_f>::value) ?
                2048
                :
                t_IR_dst_f::trans_buf_cnt;

        // IR-Collection-Buffer total count (internal Lx-cache-resident buffer of 16ki bytes)
        static const size_t m_IR_Coll_CountBuffer =
                (std::is_same<t_IR_src_f, uncompr_f>::value) ?
                2048
                :
                t_IR_src_f::trans_buf_cnt;

        // max. #elements used as upper bound to trigger transform_IR_buffer() -> "We always process 2048 elements until we transform the buffer"
        static const size_t totalProcessingCount = 2048;
    private:
        // Morph-Buffer allocation with some extra space to allow overflows
        MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value) base_t m_Morph_StartBuffer[
                m_Morph_CountBuffer + vector_element_count::value - 1
        ];
        size_t m_Morph_Count;

        // IR-Transformation-Buffer (exact) allocation + init all values to 0
        MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value) base_t m_IR_Trans_StartBuffer[m_IR_Trans_CountBuffer] = { };
        // store current starting position in IR-Trans-Buffer -> this is used as a starting point when transforming BM->PL
        base_t m_IR_Trans_StartingPos;

        // IR-Collection-Buffer (exact) allocation
        MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value) base_t m_IR_Coll_StartBuffer[m_IR_Coll_CountBuffer];
    protected:
        // Morph-Buffer current-pointer + end-pointer
        base_t * m_Morph_Buffer_CurPtr;
        base_t * const m_Morph_Buffer_EndPtr;

        // IR-Collection-Buffer current-pointer
        base_t * m_IR_Coll_Buffer_CurPtr;

        // #valid-elements that need to be transformed  in IR-Collection-Buffer, incremented in write()-function of the specific selective- & nonselective write iterators
        size_t m_IR_Coll_Count;

        // this variable gets incremented in update() every time t_op<> executes its apply(), i.e. its specific processing unit
        size_t currentProcessingCount;

        void reset_IR_buffers(){
            // IR-Collection-Buffer
            m_IR_Coll_Count = 0;
            m_IR_Coll_Buffer_CurPtr = m_IR_Coll_StartBuffer;

            // IR-Transformation-Buffer: set all values to 0
            std::memset(m_IR_Trans_StartBuffer, 0, m_IR_Trans_CountBuffer * sizeof(base_t)); // set all values back to 0
        }

        void compress_buffer() {
            const uint8_t * morphBuffer8 = reinterpret_cast<uint8_t *>(
                    m_Morph_StartBuffer
            );
            morph_batch<t_ve, t_format, uncompr_f>(
                    morphBuffer8, m_Morph_Out, m_Morph_CountBuffer
            );
            size_t overflow = m_Morph_Buffer_CurPtr - m_Morph_Buffer_EndPtr;
            memcpy(m_Morph_StartBuffer, m_Morph_Buffer_EndPtr, overflow * sizeof(base_t));
            m_Morph_Buffer_CurPtr = m_Morph_StartBuffer + overflow;
            m_Morph_Count += m_Morph_CountBuffer;
        }

        /**
         * @brief This function inserts the transformed elements from IR-Transformation-Buffer into
         *        the Morph-Buffer. It also triggers compress_buffer() if it reaches the capacity
         *        of the Morph-Buffer.
         *
         * @param numberElementsToInsert The number of elements that need to be inserted from
         *        IR-Transformation-Buffer.
         *        Note: this parameter counts in uint8 steps
         */
         void insert_into_morph_buffer(const size_t numberElementsToInsert){
            // starting point of IR-Transformation-Buffer
            const uint8_t * transBuffer8 = reinterpret_cast<uint8_t *>(
                    m_IR_Trans_StartBuffer
            );
            // starting point in Morphing-Buffer using uint8_t*
            uint8_t * morphBuffer8 = reinterpret_cast<uint8_t *>(
                    m_Morph_Buffer_CurPtr
            );

            // calculate remaining count in Morph-Buffer
            size_t remainingMorphCount64 = m_Morph_Buffer_EndPtr - m_Morph_Buffer_CurPtr;
            // cast count to uin8_t -> we are processing uint8_t-pointers in memcpy() + numberElementsToInsert is also uint8 ocunts
            size_t remainingMorphCount8 = convert_size<uint64_t, uint8_t>(remainingMorphCount64);

            // check if we overflow or exactly reach capacity of Morph-Buffer, if so trigger compress_buffer()
            if(numberElementsToInsert >= remainingMorphCount8) {
                // get difference, i.e. overflow
                size_t overflowCount8 = numberElementsToInsert - remainingMorphCount8;

                // fill Morph-Buffer with remainingMorphCount8 elements
                std::memcpy(morphBuffer8, transBuffer8, remainingMorphCount8);

                // update IR-Trans-pointer
                transBuffer8 += remainingMorphCount8;

                // now, Morph-Buffer is full -> compress
                compress_buffer();

                // get starting point again in Morph-Buffer (it could have changed due to compress_buffer()) -> just in case
                morphBuffer8 = reinterpret_cast<uint8_t *>(
                        m_Morph_Buffer_CurPtr
                );

                // insert remaining overflowCount8 elements from IR-Transformation-Buffer into Morph-Buffer
                std::memcpy(morphBuffer8, transBuffer8, overflowCount8);
            } else {
                // no overflow -> just insert
                std::memcpy(morphBuffer8, transBuffer8, numberElementsToInsert);
                // update morphBuffer8 pointer
                morphBuffer8 += numberElementsToInsert;
            }

            // finally, convert updated morphBuffer8 pointer back to base_t *
            m_Morph_Buffer_CurPtr = reinterpret_cast<base_t *>(morphBuffer8);
         }

        /**
         * @brief This function transforms the IR-Collection-Buffer to the IR-Transformation-Buffer
         *        using transform_IR_batch<> (IR transformation algorithms) and copies the transformed
         *        data elements into the Morph-Buffer.
         *
         * @param lastExecution This boolean denotes if it executes the last transformation to transform
         *                      and store only so far processed elements and not the whole buffer.
         *
         */
        void transform_IR_buffer(const bool lastExecution){
            // starting point of IR-Collection-Buffer
            const uint8_t * collBuffer8 = reinterpret_cast<uint8_t *>(
                    m_IR_Coll_StartBuffer
            );
            // starting point of IR-Transformation-Buffer
            const uint8_t * startTransBuffer8 = reinterpret_cast<uint8_t *>(
                    m_IR_Trans_StartBuffer
            );

            // tmp-pointer used in transform_IR_batch_t (gets incremented)
            uint8_t * tmpTransBuffer8 = reinterpret_cast<uint8_t *>(
                    m_IR_Trans_StartBuffer
            );

            // actual IR-transformation
            transform_IR_batch_t<t_vector_extension, t_IR_dst_f, t_IR_src_f>::apply(
                    collBuffer8, tmpTransBuffer8, m_IR_Coll_Count, m_IR_Trans_StartingPos
            );

            // Calculate #elements that need to be inserted into Morph-Buffer
            // PL->BM: insert the whole IR-Transformation-Buffer, i.e. 32 x 64-bit-words
            // BM->PL: check where the current pointer in IR-Transformation-Buffer and calculate difference to beginning
            const size_t numberElementsToInsert =
                    (std::is_same<t_IR_dst_f, bitmap_f<> >::value) ?
                        // if this is the last execution, we have to calculate the remaining bm-words from currentProcessingCount
                        (lastExecution) ?
                            round_up_div(currentProcessingCount, vector_base_t_granularity::value) * sizeof(base_t) // only remaining bm-elements
                            :
                            (bitmap_f<>::trans_buf_cnt * sizeof(base_t)) // whole buffer in uint8
                    :
                    (tmpTransBuffer8 - startTransBuffer8); // for PL as dest., we calculate the difference as it is not guaranteed that buffer is full

            // insert transformed elements into Morph-Buffer
            insert_into_morph_buffer(numberElementsToInsert);

            // update m_IR_Trans_StartingPos: we processed 2048 (valid or not), need this as new starting point for BM->PL transformation
            m_IR_Trans_StartingPos += 2048;

            // reset stuff
            reset_IR_buffers();
        }

        // Constructor
        write_iterator_base_IR(uint8_t * p_Out) :
                // init in the right order of declaration, otherwise we get -Werror=reorder
                m_Morph_Out(p_Out), // output to column in any format
                m_Morph_InitOut(m_Morph_Out),
                m_Morph_Count(0),
                m_IR_Trans_StartingPos(0),
                m_Morph_Buffer_CurPtr(m_Morph_StartBuffer),
                m_Morph_Buffer_EndPtr(m_Morph_StartBuffer + m_Morph_CountBuffer ),
                m_IR_Coll_Buffer_CurPtr(m_IR_Coll_StartBuffer),
                m_IR_Coll_Count(0),
                currentProcessingCount(0)
        {
            //
        }

    public:
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
         * 2. A pointer to the byte where more uncompressed data elements can
         *    be appended. If the uncompressed rest part of the column has
         *    already been started, then this points to the next byte after the
         *    uncompressed rest. Otherwise, this points to the byte where the
         *    uncompressed rest would begin.
         * 3. A pointer to the byte after the last acutally used byte (be it
         *    in the compressed main part or the uncompressed rest). This is
         *    only meant to be used for size calculations, not for appending
         *    more data elements.
         */
        std::tuple<size_t, uint8_t *, uint8_t *> done() {

            // execute last transformation before last morphing
            this->IR_trans_done();

            const size_t countLog = m_Morph_Buffer_CurPtr - m_Morph_StartBuffer;
            size_t outSizeComprByte;
            uint8_t * outAppendUncompr;
            if(countLog) {
                const size_t outCountLogCompr = round_down_to_multiple(
                        countLog, t_format::m_BlockSize
                );

                const uint8_t * buffer8 = reinterpret_cast<uint8_t *>(
                        m_Morph_StartBuffer
                );
                morph_batch<t_ve, t_format, uncompr_f>(
                        buffer8, m_Morph_Out, outCountLogCompr
                );
                outSizeComprByte = m_Morph_Out - m_Morph_InitOut;

                const size_t outCountLogRest = countLog - outCountLogCompr;
                if(outCountLogRest) {
                    m_Morph_Out = column<t_format>::create_data_uncompr_start(m_Morph_Out);
                    const size_t sizeOutLogRest =
                            uncompr_f::get_size_max_byte(outCountLogRest);
                    memcpy(
                            m_Morph_Out,
                            m_Morph_StartBuffer + outCountLogCompr,
                            sizeOutLogRest
                    );
                    m_Morph_Out += sizeOutLogRest;
                    outAppendUncompr = m_Morph_Out;
                }
                else
                    outAppendUncompr = column<t_format>::create_data_uncompr_start(m_Morph_Out);

                m_Morph_Count += countLog;
            }
            else {
                outSizeComprByte = m_Morph_Out - m_Morph_InitOut;
                outAppendUncompr = column<t_format>::create_data_uncompr_start(m_Morph_Out);
            }

            return std::make_tuple(
                    outSizeComprByte,
                    outAppendUncompr,
                    m_Morph_Out
            );
        }

        /**
         * @brief Returns the number of logical data elements that were stored
         * using this instance.
         * @return The number of logical data elements stored using this
         * instance.
         */
        size_t get_count_values () const {
            return m_Morph_Count;
        }

        /**
         * @brief Increments the currentProcessingCount (in IR-Transformation-Buffer) and
         *        checks if we reached the upper bound. If so, the IR-transformation
         *        from IR-Collection-Buffer to IR-Transformation-Buffer is triggered.
         *
         *        This function is primarily used in a query-operator's processing unit (t_op<>)
         *        whenever a resulting mask is calculated, more specifically in
         *        t_op_vector<t_ve, t_extra_args ...>::apply() at the very end.
         */
         void update() {
             // increment count by the number of processed elements, i.e. number of bits in the resultMask of an operator
            currentProcessingCount += vector_element_count::value;

             // check if we reached capacity to execute IR-transformation
             if(currentProcessingCount >= totalProcessingCount) {
                 this->transform_IR_buffer(false);
                 // reset counter to 0
                 currentProcessingCount = 0;
             }
         }

        /**
         * @brief This function makes sure that all possibly buffered data in IR-Collection-Buffer
         *       is transformed and stored to the output, i.e. Morph-Buffer.
         *
         *       This function should always be called before the Morph-Buffer executes its done()
         *       function.
         *
         */
        void IR_trans_done() {
            // execute only if we processed elements
            if(currentProcessingCount){
                this->transform_IR_buffer(true);
            }
        }
    };

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
    template<
            class t_vector_extension,
            class t_format,
            class t_IR_dst_f = uncompr_f,
            class t_IR_src_f = uncompr_f,
            typename std::enable_if_t<
                    // enable only if (both formats == uncompr_f) OR (both are IR-types)
                    (std::is_same<t_IR_src_f, uncompr_f>::value && std::is_same<t_IR_dst_f, uncompr_f>::value) || // to support existing implementations
                    (is_intermediate_representation_t<t_IR_src_f>::value && is_intermediate_representation_t<t_IR_dst_f>::value)
            , int> = 0

    >
    class selective_write_iterator_IR :
            public write_iterator_base_IR<t_vector_extension, t_format, t_IR_dst_f, t_IR_src_f>
    {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

    public:
        selective_write_iterator_IR(uint8_t * p_Out) :
                write_iterator_base_IR<t_vector_extension, t_format, t_IR_dst_f, t_IR_src_f>(p_Out)
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
            >(this->m_IR_Coll_Buffer_CurPtr, p_Data, p_Mask);
            this->m_IR_Coll_Buffer_CurPtr += p_MaskPopCount;
            // update m_IR_Coll_Count:
            this->m_IR_Coll_Count += p_MaskPopCount;
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
    };

    /**
     * @brief The interface for writing compressed data non-selectively.
     *
     * The default implementation buffers the appended data elements in
     * uncompressed form first. When the internal buffer is full, then the
     * batch-level morph-operator is used to compress the buffer and append the
     * compresed data to the output column's data buffer. **This interface does
     * not need to be implemented for new formats**, since the default
     * implementation can always be used as long as there is a specialization
     * of the batch-level morph-operator for the respective output format.
     */
    template<
            class t_vector_extension,
            class t_format,
            class t_IR_dst_f = uncompr_f,
            class t_IR_src_f = uncompr_f,
            typename std::enable_if_t<
                    // enable only if (both formats == uncompr_f) OR (both are IR-types)
                    (std::is_same<t_IR_src_f, uncompr_f>::value && std::is_same<t_IR_dst_f, uncompr_f>::value) || // to support existing implementations
                    (is_intermediate_representation_t<t_IR_src_f>::value && is_intermediate_representation_t<t_IR_dst_f>::value)
            , int> = 0
    >
    class nonselective_write_iterator_IR :
            public write_iterator_base_IR<t_vector_extension, t_format, t_IR_dst_f, t_IR_src_f>
    {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)

    public:
        nonselective_write_iterator_IR(uint8_t * p_Out) :
                write_iterator_base_IR<t_vector_extension, t_format, t_IR_dst_f, t_IR_src_f>(p_Out)
        {
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
            vectorlib::store<
                    t_ve,
                    vectorlib::iov::ALIGNED,
                    vector_base_t_granularity::value
            >(this->m_IR_Coll_Buffer_CurPtr, p_Data);
            this->m_IR_Coll_Buffer_CurPtr += vector_element_count::value;
            // update m_IR_Coll_Count:
            this->m_IR_Coll_Count += vector_element_count::value;
        }
    };
}

#endif //MORPHSTORE_CORE_MORPHING_WRITE_ITERATOR_IR_H
