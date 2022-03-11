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
 * @file transformation_algorithms.h
 * @brief Specific transformation algorithms (partial & full template specializations) to change
 *        logical representation of (un)compressed intermediate results, i.e.
 *             (i)  position_list -> bitmap
 *             (ii) bitmap       -> position_list
 *
 *        -> scalar + vectorized.
 *
 *        -> used to enable an unified processing approach, i.e. mixed processing strategy.
 *
 */

#ifndef MORPHSTORE_CORE_MORPHING_INTERMEDIATES_TRANSFORMATIONS_TRANSFORMATION_ALGORITHMS_H
#define MORPHSTORE_CORE_MORPHING_INTERMEDIATES_TRANSFORMATIONS_TRANSFORMATION_ALGORITHMS_H

#include <core/morphing/intermediates/transformations/interfaces.h>

namespace morphstore {

    // ************************************************************************
    // Uncompressed IR-Transformation (batch-level)
    // ************************************************************************

    // bitmap (src) -> position_list (dest)
    // partial template specialization for vectorized processing
    // general idea from: https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2018/03/07/simdbitmapdecode.c
    template<class VectorExtension>
    struct transform_IR_batch_t<VectorExtension, position_list_f<>, bitmap_f<> > {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countLog,
                uint64_t startingPos = 0
        ) {
            (void) startingPos; // TODO

            const base_t * p_BmPtr = reinterpret_cast<const base_t *>(in8);
            base_t * p_OutPtr = reinterpret_cast<base_t *>(out8);

            vector_t baseVec = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(0);
            vector_t add_v_base_t_granularity = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(
                    vector_base_t_granularity::value);
            vector_t add_v_element_count = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(vector_element_count::value);
            const base_t mask = bitmap_lookup_tables<vector_element_count::value, base_t>::mask;

            for(size_t i = 0; i < countLog; ++i) {
                // get current encoded bitmap word
                base_t word = p_BmPtr[i];

                if(!word){
                    // word is 0 -> skip
                    baseVec = vectorlib::add<VectorExtension>::apply(baseVec, add_v_base_t_granularity);
                }
                else
                {
                    for(size_t j = 0; j < ( vector_base_t_granularity::value / vector_element_count::value) ; ++j){

                        // TODO: more performance if we process 2 nibbles of sub_word concurrently ?
                        //  + Think about a vectorized version, i.e. do not materialize the whole bitmap column into
                        //    posCol, instead just pass a sub set... like decompress_and_process_batch<...> does
                        base_t sub_word = word & mask;
                        word >>= vector_element_count::value;

                        if(sub_word){
                            vector_t vec_pos = vectorlib::load<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(
                                    bitmap_lookup_tables<vector_element_count::value, base_t>::get_positions(sub_word - 1) // -1 because no entry for 0 in lookupTables ("one index before actual number")
                            );

                            //const base_t advance = __builtin_popcountll(sub_word);
                            // possible optimization: static lookup table for advance-value that returns number of matches
                            const base_t advance = vectorlib::count_matches<VectorExtension>::apply(sub_word);

                            vec_pos = vectorlib::add<VectorExtension>::apply(baseVec, vec_pos);
                            baseVec = vectorlib::add<VectorExtension>::apply(baseVec, add_v_element_count);

                            // use vectorlib::iov::UNALIGNED -> otherwise Segmentation Fault (alignment issues)
                            vectorlib::store<VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>(p_OutPtr, vec_pos);
                            // increment pointer according to the number of set bits from popcount
                            p_OutPtr += advance;
                        }
                        else
                        {
                            // sub_word is 0 -> skip
                            baseVec = vectorlib::add<VectorExtension>::apply(baseVec, add_v_element_count);
                        }
                    }
                }
            }
            out8 = reinterpret_cast<uint8_t *>(p_OutPtr);
        }
    };

    // bitmap (src) -> position_list (dest)
    // full template specialization for scalar processing
    template<>
    struct transform_IR_batch_t<scalar<v64<uint64_t>>, position_list_f<>, bitmap_f<> > {
        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countLog,
                uint64_t startingPos = 0
        ) {
            using scalar_type = typename scalar<v64<uint64_t>>::vector_t;

            const scalar_type * p_BmPtr = reinterpret_cast<const scalar_type *>(in8);
            scalar_type * p_OutPtr = reinterpret_cast<scalar_type *>(out8);
            const size_t wordBitSize = sizeof(uint64_t) << 3;

            // get current position
            uint64_t cur_pos = startingPos;

            for(size_t i = 0; i < countLog; ++i) {
                // get current encoded bitmap word
                scalar_type bm_word = p_BmPtr[i];

                uint64_t trailingZeros = 0ULL;

                while(bm_word) {
                    // count trailing zeros
                    trailingZeros = __builtin_ctzll(bm_word);
                    cur_pos += trailingZeros;

                    // insert position value
                    *p_OutPtr = cur_pos;

                    // increment variables
                    p_OutPtr++;
                    cur_pos++;
                    trailingZeros++;

                    // shift trailingZeros-bits out of the current word
                    // shifting it by more than or equal to 64 results in undefined behavior -> manually setting it to 0
                    // e.g. when uint64_t n = 2^63 -> binary = 10.......00 -> trailingZeros = 63, gets incremented then 64
                    bm_word = (trailingZeros ^ wordBitSize) ?
                            bm_word >> trailingZeros :
                            0;
                }
            }
            out8 = reinterpret_cast<uint8_t *>(p_OutPtr);
        }
    };

    // TODO: vectorized implementation (uses scalar instead)
    // position_list (src) -> bitmap (dest)
    // partial template specialization for vectorized processing
    template<class VectorExtension>
    struct transform_IR_batch_t<VectorExtension, bitmap_f<>, position_list_f<> > {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countLog,
                uint64_t startingPos = 0
        ) {
            // just call the scalar version as we do not yet support vectorized processing here
            transform_IR_batch< scalar<v64<uint64_t>>, bitmap_f<>, position_list_f<> >(
                    in8, out8, countLog, startingPos
            );
        }
    };

    // position_list (src) -> bitmap (dest)
    // full template specialization for scalar processing
    template<>
    struct transform_IR_batch_t<scalar<v64<uint64_t>>, bitmap_f<>, position_list_f<> > {
        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countLog,
                uint64_t startingPos = 0
        ) {
            using scalar_type = typename scalar<v64<uint64_t>>::vector_t;

            // to satisfy compiler unused error: just ignort startingPos as this is not relevant for PL->BM
            (void) startingPos;

            const scalar_type * p_inPos = reinterpret_cast<const scalar_type *>(in8);
            scalar_type * p_OutBm = reinterpret_cast<scalar_type *>(out8);
            const size_t wordBitSize = sizeof(uint64_t) << 3;

            for(size_t i = 0; i < countLog; ++i) {
                // get current rid
                const scalar_type rid = p_inPos[i];

                // get the index within the total bitmap (bitmap consists of 64-bit words)
                const scalar_type bm_index = rid / wordBitSize;

                // get the offset within the bitmap encoded word
                const scalar_type word_index = rid & (wordBitSize-1);

                // set bit at calculated position
                p_OutBm[bm_index] |= 1ULL << word_index;
            }

            out8 = reinterpret_cast<uint8_t *>(p_OutBm);
        }
    };

    // ************************************************************************
    // Uncompressed IR-Transformation (column-level)
    // ************************************************************************

    // bitmap (src) -> position_list (dest)
    template<class t_vector_extension>
    struct transform_IR_t<t_vector_extension, position_list_f<>, bitmap_f<> > {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)

        static
        const column< position_list_f<uncompr_f> > *
        apply(
                const column< bitmap_f<uncompr_f> > * inBm
        ) {
            const size_t countLog = inBm->get_count_values();
            const uint8_t * in8 = inBm->get_data();

            // pessimistic allocation: assuming every bit is set in bitmap
            // TODO: Think about a more space-efficient way (e.g. using metadata with popcount of bitmap)
            auto outPos = new column< position_list_f<uncompr_f> >(
                    countLog * vector_base_t_granularity::value * sizeof(base_t)
            );

            uint8_t * out8 = outPos->get_data();
            const uint8_t * const initOut8 = out8;

            // IR-transformation
            transform_IR_batch<t_vector_extension, position_list_f<>, bitmap_f<> >(
                    in8, out8, countLog, 0
            );

            // converting "back" to uint64_t elements -> we processed it with uint8_t *
            const size_t outPosCount = convert_size<uint8_t, uint64_t>(out8 - initOut8);
            outPos->set_meta_data(outPosCount, outPosCount * sizeof(base_t));

            return outPos;
        }
    };

    // position_list (src) -> bitmap (dest)
    template<class t_vector_extension>
    struct transform_IR_t<t_vector_extension, bitmap_f<> , position_list_f<> > {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)

        static
        const column< bitmap_f<uncompr_f> > *
        apply(
                const column< position_list_f<uncompr_f> > * inPos
        ) {
            const size_t countLog = inPos->get_count_values();
            const uint8_t * in8 = inPos->get_data();

            //  bitmap column allocation:
            //  (1) get the highest position value from inPos: inPos[countLog-1] -> Assuming positions are ordered ASC
            //  (2) bm count = round_up_div(highest_pos / 64 ); if base_t is uint64_t (granularity)
            const base_t high_pos_val = reinterpret_cast<const base_t *>(in8)[countLog-1];
            const size_t bm_count = round_up_div(high_pos_val, vector_base_t_granularity::value);
            auto outBm = new column< bitmap_f<uncompr_f> >(
                    bm_count * sizeof(base_t)
            );

            uint8_t * out8 = outBm->get_data();

            // IR-transformation
            transform_IR_batch<t_vector_extension, bitmap_f<>, position_list_f<> >(
                    in8, out8, countLog
            );

            outBm->set_meta_data(bm_count, bm_count * sizeof(base_t));

            return outBm;
        }
    };

    // ************************************************************************
    // Compressed IR-Transformation (column-level)
    // ************************************************************************

    // TODO: write code + enable testing
}

#endif //MORPHSTORE_CORE_MORPHING_INTERMEDIATES_TRANSFORMATIONS_TRANSFORMATION_ALGORITHMS_H