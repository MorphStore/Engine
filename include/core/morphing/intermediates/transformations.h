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
 * @file transformations.h
 * @brief Change logical representation of (un)compressed intermediate results, i.e.
 *             (i)  position_list -> bitmap
 *             (ii) bitmap       -> position_list
 *
 *        Used to enable an unified processing approach, i.e. mixed processing strategy.
 *
 */

#ifndef MORPHSTORE_CORE_MORPHING_INTERMEDIATES_TRANSFORMATIONS_H
#define MORPHSTORE_CORE_MORPHING_INTERMEDIATES_TRANSFORMATIONS_H

#include <core/morphing/intermediates/position_list.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/morphing/intermediates/representation.h>
#include <core/morphing/morph_batch.h>
#include <core/morphing/morph.h>

namespace morphstore {

    /**
     * @brief A template specialization of the morph-operator handling the case
     *        when the source and the destination IR-types are the same and their
     *        inner-formats are uncompressed.
     *
     *        We need to make this case explicit, since otherwise, the choice of the
     *        right partial template specialization is ambiguous for the compiler.
     */

    // bitmap
    template<class t_vector_extension>
    struct morph_t<t_vector_extension, bitmap_f<>, bitmap_f<>> {
        static
        const column< bitmap_f<> > *
        apply(const column< bitmap_f<> > * inCol) {
            return inCol;
        };
    };

    // position-list
    template<class t_vector_extension>
    struct morph_t<t_vector_extension, position_list_f<>, position_list_f<>> {
        static
        const column< position_list_f<> > *
        apply(const column< position_list_f<> > * inCol) {
            return inCol;
        };
    };


    // ************************************************************************
    // Morph-operators (batch-level) - uncompressed IR-Transformation
    // ************************************************************************

    // bitmap (src) -> position_list (dest)
    // partial template specialization for vectorized processing
    // general idea from: https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2018/03/07/simdbitmapdecode.c
    // TODO: change this to work on a batch-basis:
    //       bm -> pl: need last pl word + allocate as many position integer as pop_count of current bitmap word
    template<class VectorExtension>
    struct morph_batch_t<VectorExtension, position_list_f<>, bitmap_f<> > {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countLog
        ) {
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
    struct morph_batch_t<scalar<v64<uint64_t>>, position_list_f<>, bitmap_f<> > {
    static void apply(
            const uint8_t * & in8,
            uint8_t * & out8,
            size_t countLog
    ) {
        using scalar_type = typename scalar<v64<uint64_t>>::vector_t;

        const scalar_type * p_BmPtr = reinterpret_cast<const scalar_type *>(in8);
        scalar_type * p_OutPtr = reinterpret_cast<scalar_type *>(out8);
        const size_t wordBitSize = sizeof(uint64_t) << 3;

        for(size_t i = 0; i < countLog; ++i) {
            // get current encoded bitmap word
            scalar_type word = p_BmPtr[i];

            const size_t offset = i * wordBitSize;
            uint64_t trailingZeros, pos = 0ULL;

            while(word) {
                // count trailing zeros
                trailingZeros = __builtin_ctzll(word);
                pos += trailingZeros;

                // insert position value
                *p_OutPtr = pos + offset;

                // increment variables
                p_OutPtr++;
                pos++;
                trailingZeros++;

                // shift trailingZeros-bits out of the current word
                // shifting it by more than or equal to 64 results in undefined behavior -> manually setting it to 0
                // e.g. when uint64_t n = 2^63 -> binary = 10.......00 -> trailingZeros = 63, gets incremented then 64
                word = (trailingZeros ^ wordBitSize) ?
                       word >> trailingZeros :
                       0;
            }
        }
        out8 = reinterpret_cast<uint8_t *>(p_OutPtr);
    }
    };

    // TODO: vectorized implementation missing: => use scalar instead
    // position_list (src) -> bitmap (dest)
    // partial template specialization for vectorized processing
    template<class VectorExtension>
    struct morph_batch_t<VectorExtension, bitmap_f<>, position_list_f<> > {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countLog
        ) {
            morph_batch< scalar<v64<uint64_t>>, bitmap_f<>, position_list_f<> >(in8, out8, countLog);
        }
    };

    // position_list (src) -> bitmap (dest)
    // full template specialization for scalar processing
    template<>
    struct morph_batch_t<scalar<v64<uint64_t>>, bitmap_f<>, position_list_f<> > {
    static void apply(
            const uint8_t * & in8,
            uint8_t * & out8,
            size_t countLog
    ) {
        using scalar_type = typename scalar<v64<uint64_t>>::vector_t;

        const scalar_type * p_inPos = reinterpret_cast<const scalar_type *>(in8);
        scalar_type * p_OutBm = reinterpret_cast<scalar_type *>(out8);
        const size_t wordBitSize = sizeof(uint64_t) << 3;

        uint64_t word = 0;
        size_t cur_upper_bound = wordBitSize;

        for(size_t i = 0; i < countLog; ++i) {
            // get current rid
            const scalar_type rid = p_inPos[i];

            // check if the current row-id is in the current word boundary
            if(rid < cur_upper_bound) {
                // shift row-id-index into current word
                word |= 1ULL << (rid & (wordBitSize-1));
            }
                // current row-id does not fit into current word + inserting filling zeros
            else
            {
                // store current word and update pointers/counters
                *p_OutBm = word;
                ++p_OutBm;
                word = 0;
                cur_upper_bound += wordBitSize;

                // insert filling 0s if necessary to keep bitmap semantic correct
                while(rid >= cur_upper_bound){
                    *p_OutBm = word;
                    ++p_OutBm;
                    word = 0;
                    cur_upper_bound += wordBitSize;
                }

                // finally, add number to current bitmap-word
                word |= 1ULL << (rid & (wordBitSize-1));
            }
        }

        // eventually store word in bitmaps (e.g. if we process less than 64 elements)
        if(word){
            *p_OutBm = word;
            ++p_OutBm;
        }

        out8 = reinterpret_cast<uint8_t *>(p_OutBm);
    }
    };

    // ************************************************************************
    // Morph-operators (column-level) - uncompressed IR-Transformation
    // ************************************************************************

    // bitmap (src) -> position_list (dest)
    template<class t_vector_extension>
    struct morph_t<t_vector_extension, position_list_f<>, bitmap_f<> > {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)

        static
        const column< position_list_f<uncompr_f> > *
        apply(
                const column< bitmap_f<uncompr_f> > * inCol
        ) {
            const size_t countLog = inCol->get_count_values();
            const uint8_t * in8 = inCol->get_data();

            // pessimistic allocation: assuming every bit is set in bitmap
            // TODO: Think about a more space-efficient way
            auto outCol = new column< position_list_f<uncompr_f> >(
                    countLog * vector_base_t_granularity::value * sizeof(base_t)
            );

            uint8_t * out8 = outCol->get_data();
            const uint8_t * const initOut8 = out8;

            morph_batch<t_vector_extension, position_list_f<>, bitmap_f<> >(
                    in8, out8, countLog
            );

            // converting "back" to uint64_t elements -> we processed it with uint8_t *
            const size_t outPosCount = convert_size<uint8_t, uint64_t>(out8 - initOut8);
            outCol->set_meta_data(outPosCount, outPosCount * sizeof(base_t));

            return outCol;
        }
    };

    // position_list (src) -> bitmap (dest)
    template<class t_vector_extension>
    struct morph_t<t_vector_extension, bitmap_f<> , position_list_f<> > {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)

        static
        const column< bitmap_f<uncompr_f> > *
        apply(
                const column< position_list_f<uncompr_f> > * inCol
        ) {
            const size_t countLog = inCol->get_count_values();
            const uint8_t * in8 = inCol->get_data();

            //  bitmap column allocation:
            //  (1) get the highest position value from inCol: inCol[countLog-1] -> Assuming positions are ordered ASC
            //  (2) bm count = round_up_div(highest_pos / 64 ); if base_t is uint64_t (granularity)
            const base_t high_pos_val = reinterpret_cast<const base_t *>(in8)[countLog-1];
            const size_t bm_count = round_up_div(high_pos_val, vector_base_t_granularity::value);
            auto outCol = new column< bitmap_f<uncompr_f> >(
                    bm_count * sizeof(base_t)
            );

            uint8_t * out8 = outCol->get_data();
            const uint8_t * const initOut8 = out8;

            morph_batch<t_vector_extension, bitmap_f<>, position_list_f<> >(
                    in8, out8, countLog
            );

            // converting "back" to uint64_t elements -> we processed it with uint8_t *
            const size_t outPosCount = convert_size<uint8_t, uint64_t>(out8 - initOut8);
            outCol->set_meta_data(outPosCount, outPosCount * sizeof(base_t));

            return outCol;
        }
    };

    // ************************************************************************
    // Morph-operators (column-level) - compressed IR-Transformation
    // ************************************************************************

    template<
            class t_vector_extension,
            class t_IR_dest_inner_f,
            class t_IR_src_inner_f
    >
    struct morph_t<t_vector_extension, position_list_f<t_IR_dest_inner_f>, bitmap_f<t_IR_src_inner_f> >{
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)

        using t_IR_src_f = bitmap_f<t_IR_src_inner_f>;
        using t_IR_dest_f = position_list_f<t_IR_dest_inner_f>;

        static
        const column< t_IR_dest_f > *
        apply(
                const column< t_IR_src_f > * inCol
        ) {
            // (1) decompression of bitmap
            auto inCol_decompr =
                    morph_t<
                        t_vector_extension,
                        bitmap_f<uncompr_f>,
                        bitmap_f<typename t_IR_src_f::t_inner_f>
                    >::apply(inCol);

            // (2) IR-transformation (uncompressed): bm -> pl
            auto pl_decompr =
                    morph_t<
                        t_vector_extension,
                        position_list_f<>,
                        bitmap_f<>
                    >::apply(inCol_decompr);

            // (3) compression of position-list
            auto outCol =
                    morph_t<
                        t_vector_extension,
                        position_list_f<typename t_IR_dest_f::t_inner_f>,
                        position_list_f<uncompr_f>
                    >::apply(pl_decompr);

            return outCol;
        }
    };

    // compressed-position-list (src) -> compressed-bitmap (dest)
    template<
            class t_vector_extension,
            class t_IR_dest_inner_f,
            class t_IR_src_inner_f
    >
    struct morph_t<t_vector_extension, bitmap_f<t_IR_dest_inner_f>, position_list_f<t_IR_src_inner_f> >{
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)

        using t_IR_src_f = position_list_f<t_IR_src_inner_f>;
        using t_IR_dest_f = bitmap_f<t_IR_dest_inner_f>;

        static
        const column< t_IR_dest_f > *
        apply(
                const column< t_IR_src_f > * inCol
        ) {
            // (1) decompression of position-list
            auto inCol_decompr =
                    morph_t<
                        t_vector_extension,
                        position_list_f<uncompr_f>,
                        position_list_f<typename t_IR_src_f::t_inner_f>
                    >::apply(inCol);

            // (2) IR-transformation (uncompressed): pl -> bm
            auto bm_decompr =
                    morph_t<
                        t_vector_extension,
                        bitmap_f<>,
                        position_list_f<>
                    >::apply(inCol_decompr);

            // (3) compression of bitmap
            auto outCol =
                    morph_t<
                        t_vector_extension,
                        bitmap_f<typename t_IR_dest_f::t_inner_f>,
                        bitmap_f<uncompr_f>
                    >::apply(bm_decompr);

            return outCol;
        }
    };

}

#endif //MORPHSTORE_CORE_MORPHING_INTERMEDIATES_TRANSFORMATIONS_H