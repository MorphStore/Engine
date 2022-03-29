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
 * @file wah.h
 * @brief 32-bit Word-Aligned Hybrid (WAH) compression format for bitmaps. Generally 32-bit encoding + RLE encoding.
 *
 *        TODO: decompress_and_process_batch + add 64-bit support for WAH compression
 */

#ifndef MORPHSTORE_CORE_MORPHING_WAH_H
#define MORPHSTORE_CORE_MORPHING_WAH_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <core/morphing/write_iterator_IR.h>
#include <vector/vector_primitives.h>
#include <vector/vector_extension_structs.h>

#include <iostream>
#include <cstring>

namespace morphstore {

    // ************************************************************************
    // Format
    // ************************************************************************
    struct wah_f : public format {

        static size_t get_size_max_byte(size_t p_CountValues) {
            return convert_size<uint64_t, uint8_t>(p_CountValues);
        }

        // TODO: think about the blockSize for wah_f -> for simplicity, assuming 1 to get write_iterator working...
        static const size_t m_BlockSize = 1;
    };


    // global wah-processing state
    struct wah32_processing_state_t {
        size_t numberOnes;
        size_t numberZeros;
        //uint32_t activeWord; TODO: add?

        wah32_processing_state_t(size_t ones, size_t zeros) : numberOnes(ones), numberZeros(zeros)
        {
        }

        // important constants used in compression / decompression:
        static const uint32_t ALL_ONES = 0x7FFFFFFF; // 0111 1111 ... -> all ones except MSB
        static const uint32_t ONE_FILL_HEADER_MASK = 0xC0000000;  // 11.. ....
        static const uint32_t ZERO_FILL_HEADER_MASK = 0x80000000; // 10... ....
        static const uint32_t FILL_WORD_COUNT_MASK = 0x3FFFFFFF; // all bits except MSB and MSB+1
    };

    // ************************************************************************
    // Morph-operators (batch-level)
    // ************************************************************************

    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------

    /**
     * @brief This is the primary batch-oriented compression template as we need a temporary
     *        state (wah_processing_state_t) and the current morph_batch-interface does not allow
     *        to include any additional state in its input parameters.
     *
     *        => Similar to delta_f
     *
     */
    template<class t_vector_extension>
    struct wah_compress_batch_with_state_t{
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

        // this function returns the following 31-bits from a pointer using a starting index
        static uint32_t getNext(size_t startingIndex, const uint32_t *& inBase32, const uint32_t *& endInBase32){
            // Using 64-bit words to shift overlapping stuff in first half & second half
            uint64_t result = 0;
            uint64_t l1 = 0;
            uint64_t l2 = 0;
            uint64_t l = 0;
            // index in bitmap of 32-bit words, the i-th 32-bit word...
            size_t index = startingIndex / 32;
            // offset within current 32-bit word
            const size_t offset = startingIndex % 32;

            // get value and increment index
            l1 = inBase32[index++];
            // check if we are out of bound, if not fetch next value, otherwise use just 0 value
            if(inBase32+index <= endInBase32) l2 = inBase32[index];
            // divide numbers into one 64-bit word
            l = (l1 << 32) + l2;
            // shift by current index and fetch only the first 31-bits (mask: 0x7FFFFFFF)
            result = (l >> (33 - offset)) & wah32_processing_state_t::ALL_ONES;

            return (uint32_t) result;
        }

        // function adds literal to output and increments pointer
        static void addLiteral(uint32_t *& outBase32, uint32_t number){
            *outBase32 = number;
            ++outBase32;
        }

        // this function creates a 1-fill word and appends it to the output
        static void flushOnes(uint32_t *& outBase32, size_t & counter){
            if(counter > 0){
                // 1-Fill = 11.. .... + counter
                uint32_t one_fill = wah32_processing_state_t::ONE_FILL_HEADER_MASK + counter;
                // reset counter
                counter = 0;
                // add to output and increment pointer
                *outBase32 = one_fill;
                ++outBase32;
            }
        }

        // this function creates a 0-fill word and appends it to the output
        static void flushZeros(uint32_t *& outBase32, size_t & counter){
            if(counter > 0){
                // 0-Fill = 10.. .... + counter
                uint32_t zero_fill = wah32_processing_state_t::ZERO_FILL_HEADER_MASK + counter;
                // reset counter
                counter = 0;
                // add to output and increment pointer
                *outBase32 = zero_fill;
                ++outBase32;
            }
        }

        // this function flushes remaining 1-Fills & 0-Fill (if any)
        static void done(uint8_t * & outBase8, wah32_processing_state_t & wahProcessingState) {
            uint32_t *outBase32 = reinterpret_cast<uint32_t *>(outBase8);
            flushOnes(outBase32, wahProcessingState.numberOnes);
            flushZeros(outBase32, wahProcessingState.numberZeros);
            outBase8 = reinterpret_cast<uint8_t *>(outBase32);
        }

        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countInBase64,
                wah32_processing_state_t & wahProcessingState
        ) {
            const size_t countInBase32 = countInBase64 * 2;
            const uint32_t *inBase32 = reinterpret_cast<const uint32_t *>(in8);
            const uint32_t *endInBase32 = inBase32 + countInBase32;
            uint32_t *outBase32 = reinterpret_cast<uint32_t *>(out8);

            // General processing:
            //  (1) take 31 bits from input
            //  (2) if all zero (0-fill) -> increment zero-counter, flush ones
            //  (3) if all ones (1-fill) -> increment ones-counter, flush zeros
            //  (4) else literal         -> flush ones, flush zeros, add literal

            // we iterate through a portion of bits from the input uncompressed bitmap (which is encoded in 64-bit words)
            const size_t countInBits = countInBase32 << 5;

            for (size_t i = 0; i < countInBits;) {
                // (1) get next 31-bits
                uint32_t currentNumber = getNext(i, inBase32, endInBase32);
                // increment counter
                i += 31;

                if (currentNumber == 0) {
                    // case (2) all zeros
                    ++wahProcessingState.numberZeros;
                    // write 1-Fills (if any) to output
                    flushOnes(outBase32, wahProcessingState.numberOnes);
                } else if (currentNumber == wah32_processing_state_t::ALL_ONES) {
                    // case (3) all ones
                    ++wahProcessingState.numberOnes;
                    // write 0-Fills (if any) to output
                    flushZeros(outBase32, wahProcessingState.numberZeros);
                } else {
                    // case (4) literals
                    flushOnes(outBase32, wahProcessingState.numberOnes);
                    flushZeros(outBase32, wahProcessingState.numberZeros);
                    addLiteral(outBase32, currentNumber);
                }
            }
            out8 = reinterpret_cast<uint8_t *>(outBase32);
        }
    };

    template<class t_vector_extension>
    struct morph_batch_t<
            t_vector_extension,
            wah_f,
            uncompr_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countInBase64
        ) {
            wah32_processing_state_t wahProcessingState(0,0);
            wah_compress_batch_with_state_t<t_ve>::apply(in8, out8, countInBase64, wahProcessingState);
            // eventually flush remaining zeros and ones
            wah_compress_batch_with_state_t<t_ve>::done(out8, wahProcessingState);
        }
    };

    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------

    // TODO: include processing state (for decompress_and_process_batch)
    /**
     * @brief Idea: Use internal buffer in which chunks of 31-bits are written to:
     *              - countLog specifies how many WAH-words are processed:
     *                  if literal: decrement countLog
     *                  if fill: countLog only gets decremented if the counter
     *                           of the fill reaches 0
     *              - assuming the output pointer is allocated with enough
     *                memory
     *
     */
    template<class t_vector_extension>
    struct wah_decompress_batch_with_state_t {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

        static void write_buffer_31Bits(uint32_t * buffer, uint32_t word, size_t pos){
            // index of 32-bit words, i.e. the i-th 32-bit word
            size_t index = pos / 32;
            // offset within current 32-bit word
            const size_t offset = pos % 32;

            // use uin64_t to store overlapping numbers, i.e. if it splits
            uint64_t tmp = ((uint64_t)buffer[index] << 32) + buffer[index + 1];
            tmp |= (uint64_t)word << (33 - offset);

            buffer[index] = (uint32_t)(tmp >> 32);
            buffer[index + 1] = (uint32_t)(tmp & 0xFFFFFFFF);
        }

        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countWAHWords
        ) {
            const uint32_t *in32 = reinterpret_cast<const uint32_t *>(in8);
            uint32_t *out32 = reinterpret_cast<uint32_t *>(out8);

            // tmp-buffer => if this buffer is full, we flush it to the output, and reset buffer to beginning => to avoid overflow
            // buffer is initialized with 0s (we shift 31-bit chunks into it) => static allocation 310 x 32-bit = 1240 byte
            const size_t elementCount = 310;
            uint32_t tmp[elementCount] = { };
            // index to track the capacity within tmp-buffer
            size_t tmp_curPos = 0;
            const size_t tmp_endPos = (32 * elementCount) - 1;

            // we return when countWAHWords reached 0
            while(countWAHWords) {
                // get current WAH-encoded word (32-bit)
                uint32_t currentWord = *in32;

                // check whether literal- or fill-word:
                if( (currentWord & 0x80000000) == 0){
                    // LITERAL: just add to buffer
                    write_buffer_31Bits(tmp, currentWord, tmp_curPos);

                    // update index
                    tmp_curPos += 31;

                    // check if we reached capacity of tmp-buffer
                    if(tmp_curPos >= tmp_endPos) {
                        // copy whole buffer to output
                        std::memcpy(out32, tmp, elementCount * sizeof(uint32_t));
                        out32 += elementCount;
                        // reset buffer
                        std::memset(tmp, 0, elementCount * sizeof(uint32_t));
                        tmp_curPos = 0;
                    }
                } else{
                    // FILL-WORD: add 31-bit chunks of 0s / 1s as long as counter is not 0
                    size_t fillCounter = currentWord & wah32_processing_state_t::FILL_WORD_COUNT_MASK;
                    // if 0-Fill, we just need to increment tmp_CurPos and check if capacity is reached
                    // if 1-Fill, write 31 x 1s in one loop, then increment, then check capacity
                    const bool isOneFill =
                            (currentWord & wah32_processing_state_t::ONE_FILL_HEADER_MASK)
                            == wah32_processing_state_t::ONE_FILL_HEADER_MASK;
                    while(fillCounter) {
                        if(isOneFill) {
                            // write to buffer
                            write_buffer_31Bits(tmp, wah32_processing_state_t::ALL_ONES, tmp_curPos);
                        }

                        // update index and decrement fillCounter
                        tmp_curPos += 31;
                        --fillCounter;

                        // check if we reached capacity of tmp-buffer
                        if(tmp_curPos >= tmp_endPos) {
                            // copy whole buffer to output
                            std::memcpy(out32, tmp, elementCount * sizeof(uint32_t));
                            out32 += elementCount;
                            // reset buffer
                            std::memset(tmp, 0, elementCount * sizeof(uint32_t));
                            tmp_curPos = 0;
                        }
                    }
                }

                ++in32;
                --countWAHWords;
            }
            // store remaining stuff
            if(tmp_curPos > 0) {
                const size_t remainingCount = round_up_div(tmp_curPos, 32);
                std::memcpy(out32, tmp, remainingCount * sizeof(uint32_t));
                out32 += remainingCount;
            }

            out8 = reinterpret_cast<uint8_t *>(out32);
        }
    };

    template<class t_vector_extension>
    struct morph_batch_t<
            t_vector_extension,
            uncompr_f,
            wah_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countInBase32
        ) {
            // TODO: we could remove this, but as long as decompress_and_process_batch is not implemented, we could need this (extra state etc.)
            wah_decompress_batch_with_state_t<t_ve>::apply(in8, out8, countInBase32);
        }
    };


    // ************************************************************************
    // Morph-operators (column-level)
    // ************************************************************************

    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    template<
            class t_vector_extension
    >
    struct morph_t<
            t_vector_extension,
            wah_f,
            uncompr_f
    > {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
        using out_f = wah_f;
        using in_f = uncompr_f;

        static
        const column<out_f> *
        apply(const column<in_f> * inCol) {
            // assuming input is in uint64_t words
            const size_t countBase64 = inCol->get_count_values();

            // pessimistic allocation: assuming that inCol consists only of literals
            // -> Overhead size is 1/32 = 3,125% (for every 32-bits, we have to allocate 1-bit extra)
            const size_t countBase32 = countBase64 * 2;
            const size_t overhead = round_up_div(countBase32, 32);
            const size_t estimatedCount64 = round_up_div( (countBase32 + overhead), 2);

            auto outCol = new column<wah_f>(
                    estimatedCount64 * sizeof(uint64_t)
            );

            const uint8_t* in8 = inCol->get_data();

            uint8_t * out8 = outCol->get_data();
            const uint8_t * const initOut8 = out8;

            morph_batch<t_vector_extension, wah_f, uncompr_f>(
                    in8, out8, countBase64
            );

            // converting "back" to uint32_t elements -> we processed it with uint8_t *
            const size_t outDataCount32 = convert_size<uint8_t, uint32_t>(out8 - initOut8);

            outCol->set_meta_data(outDataCount32, outDataCount32 * sizeof(uint32_t));

            return outCol;
        }
    };

    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    template<
            class t_vector_extension
    >
    struct morph_t<
            t_vector_extension,
            uncompr_f,
            wah_f
    > {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
        using out_f = uncompr_f;
        using in_f = wah_f;

        static
        const column<out_f> *
        apply(const column<in_f> * inCol) {
            // count is related to uint32_t words
            const size_t countBase32 = inCol->get_count_values();

            // TODO: pessimistic allocation is hard since we can not estimate how many bits are outputted...
            //      => for now we simply use a large number -> REMOVE THIS
            auto outCol = new column<uncompr_f>(
                    100*100*10 * sizeof(uint64_t)
            );

            const uint8_t* in8 = inCol->get_data();

            uint8_t * out8 = outCol->get_data();
            const uint8_t * const initOut8 = out8;

            morph_batch<t_vector_extension, uncompr_f, wah_f>(
                    in8, out8, countBase32
            );

            // converting "back" to uint32_t elements -> we processed it with uint8_t *
            const size_t outDataCount32 = convert_size<uint8_t, uint32_t>(out8 - initOut8);

            outCol->set_meta_data(outDataCount32, outDataCount32 * sizeof(uint32_t));

            return outCol;
        }
    };

    // ------------------------------------------------------------------------
    // Sequential write
    // ------------------------------------------------------------------------

    /**
     * @brief A specialization for `wah_f`.
     *
     *        The write-iterator calls the batch-level morph-operator for the
     *        recompression multiple times. When using the default implementation with
     *        `wah_f`, each call of the recompression thinks it is the first and,
     *        thus, do not consider the current number of zeros and ones that need to be
     *        flushed (wah-processing-state). This specialization of `write_iterator_base_IR`
     *        solves this problem by not calling the normal batch-level morph-operator for
     *        `wah_f`, but another function 'wah_compress_batch_with_state_t::apply(...)',
     *        which is also internally used by the batch-level morph-operator for `wah_f`.
     *        `wah_compress_batch_with_state_t::apply(...)` takes the current wah32_processing_state_t
     *        as an additional parameter.
     *
     *        => Note: this design decision is similar to 'delta_f'.
     *
     *        @todo It is bad to duplicate so large parts of the default
     *              implementation.
     */
    template<
            class t_vector_extension,
            class t_IR_dst_f,
            class t_IR_src_f
            // Comment-out the enable_if<...> as this leads to 'error: default template arguments may not be used in partial specializations' -> TODO: fix this
            /*typename std::enable_if_t<
                    // enable only if both are IR-types
                    (is_intermediate_representation_t<t_IR_src_f>::value && is_intermediate_representation_t<t_IR_dst_f>::value)
            , int> = 0*/
    >
    class write_iterator_base_IR<
            t_vector_extension,
            wah_f,
            t_IR_dst_f,
            t_IR_src_f
        > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

        using t_format = wah_f;

        // Morph-Buffer output-pointer
        uint8_t * m_Morph_Out;
        const uint8_t * const m_Morph_InitOut;
    public:
        // Morph-Buffer total count using 64-bit words
        static const size_t m_Morph_CountBuffer = round_up_to_multiple(
                t_format::m_BlockSize, 2048
        );

        // IR-Transformation-Buffer total count
        // In general, buffer can hold up to 2048 uncompressed data elements (internal Lx-cache-resident buffer of 16ki bytes)
        static const size_t m_IR_Trans_CountBuffer = t_IR_dst_f::trans_buf_cnt;

        // IR-Collection-Buffer total count (internal Lx-cache-resident buffer of 16ki bytes)
        static const size_t m_IR_Coll_CountBuffer = t_IR_src_f::trans_buf_cnt;

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
        // main difference to the general write_iterator_base_IR implementation: additional wah-processing-state
        wah32_processing_state_t wahProcessingState;

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
            // using customized compress template which takes the wahProcessingState as additional input parameter
            wah_compress_batch_with_state_t<t_ve>::apply(
                    morphBuffer8, m_Morph_Out, m_Morph_CountBuffer, wahProcessingState
            );
            size_t overflow = m_Morph_Buffer_CurPtr - m_Morph_Buffer_EndPtr;
            memcpy(m_Morph_StartBuffer, m_Morph_Buffer_EndPtr, overflow * sizeof(base_t));
            m_Morph_Buffer_CurPtr = m_Morph_StartBuffer + overflow;
            m_Morph_Count += convert_size<uint64_t, uint32_t>(m_Morph_CountBuffer);
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
                wahProcessingState( wah32_processing_state_t(0,0) ),
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
                /*const size_t outCountLogCompr = round_down_to_multiple(
                        countLog, t_format::m_BlockSize
                );*/

                const uint8_t * buffer8 = reinterpret_cast<uint8_t *>(
                        m_Morph_StartBuffer
                );

                // using customized compress template which takes the wahProcessingState as additional input parameter
                wah_compress_batch_with_state_t<t_ve>::apply(
                        buffer8, m_Morph_Out, countLog, wahProcessingState
                );

                // eventually flush remaining zeros and ones from wahProcessingState
                wah_compress_batch_with_state_t<t_ve>::done(m_Morph_Out, wahProcessingState);

                outSizeComprByte = m_Morph_Out - m_Morph_InitOut;

                /*const size_t outCountLogRest = countLog - outCountLogCompr;
                if(outCountLogRest) {
                    m_Morph_Out = column<t_format>::create_data_uncompr_start(m_Morph_Out);
                    const size_t sizeOutLogRest =
                            uncompr_f::get_size_max_byte(outCountLogRest);
                    memcpy(
                            m_Morph_Out,
                            m_Morph_StartBuffer + outCountLogCompr,
                            sizeOutLogRest
                    );
                    m_Morph_Out += convert_size<uint8_t, uint32_t>(sizeOutLogRest);
                    outAppendUncompr = m_Morph_Out;
                }
                else{
                    outAppendUncompr = column<t_format>::create_data_uncompr_start(m_Morph_Out);
                }*/

                m_Morph_Count += convert_size<uint8_t, uint32_t>(outSizeComprByte);
            }
            else {
                // eventually flush remaining zeros and ones from wahProcessingState
                wah_compress_batch_with_state_t<t_ve>::done(m_Morph_Out, wahProcessingState);

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
}

#endif //MORPHSTORE_CORE_MORPHING_WAH_H
