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
 *        TODO: write_iterator_base with special compress_batch() + decompress_and_process_batch + add 64-bit support
 */

#ifndef MORPHSTORE_CORE_MORPHING_WAH_H
#define MORPHSTORE_CORE_MORPHING_WAH_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
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

        //static const size_t m_BlockSize = 4;
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
    struct compress_batch_with_state_t{
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
            compress_batch_with_state_t<t_ve>::apply(in8, out8, countInBase64, wahProcessingState);
            // eventually flush remaining zeros and ones
            compress_batch_with_state_t<t_ve>::done(out8, wahProcessingState);
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
    struct decompress_batch_with_state_t {
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
                size_t countWAHWords,
                wah32_processing_state_t & wahProcessingState
        ) {
            (void) wahProcessingState;

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
                size_t countInBase64
        ) {
            wah32_processing_state_t wahProcessingState(0,0);
            decompress_batch_with_state_t<t_ve>::apply(in8, out8, countInBase64, wahProcessingState);
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
            // assuming input is in uint64_t words
            const size_t countBase64 = inCol->get_count_values();

            // TODO: pessimistic allocation is hard since we can not estimate how many bits are outputted...
            //      => for now we simply use a large number -> REMOVE THIS
            auto outCol = new column<uncompr_f>(
                    100*100*10 * sizeof(uint64_t)
            );

            const uint8_t* in8 = inCol->get_data();

            uint8_t * out8 = outCol->get_data();
            const uint8_t * const initOut8 = out8;

            morph_batch<t_vector_extension, uncompr_f, wah_f>(
                    in8, out8, countBase64
            );

            // converting "back" to uint64_t elements -> we processed it with uint8_t *
            const size_t outDataCount64 = convert_size<uint8_t, uint64_t>(out8 - initOut8);

            outCol->set_meta_data(outDataCount64, outDataCount64 * sizeof(uint64_t));

            return outCol;
        }
    };
}

#endif //MORPHSTORE_CORE_MORPHING_WAH_H
