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
 * @file select_bm_compr.h
 * @brief Select-operator based on the vector-lib, weaving the operator's core
 *        into the decompression routine of the input data's format.
 *
 *        Returns a compressed column with bitmaps as intermediate representation.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_BM_COMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_BM_COMPR_H

// @todo Include this as soon as the interfaces are harmonized.
//#include <core/operators/interfaces/select.h>
#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/morphing/format.h>
#include <core/morphing/write_iterator.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <tuple>
#include <type_traits>

#include <cstdint>

namespace morphstore {

    // If the following macro is defined, then the template parameter is a template
    // class, otherwise, it is a (specialized) class.
#undef COMPARE_OP_AS_TEMPLATE_CLASS

    template<
            class t_vector_extension,
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
            template<class, int> class t_compare,
#else
            class t_compare,
#endif
            class t_out_f
    >
    struct select_processing_unit_wit {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

        // TODO: We actually just store the bitmap non-selectivity, but at the moment there is no support for
        //       just storing a base_t element (nonselective's write() needs a vector_t p_Data as input).
        //       Instead of implementing another write function, we just use a selective_write_iterator with a mask
        //       to store the first element in vector_t ('workaround').
        //       -> In the future, we should add a function in nonselective_write_iterator that also accepts just base_t elements when filling the buffer.
        //          -> OR: fill vector register until its capacity is reached (e.g. 4-64 bitmap words in avx2), then use nonselective's write()
        struct state_t {
            const vector_t m_Predicate;
            const vector_mask_t m_write_mask = 1; // we just want to store the first word in the vector
            vector_t m_bitmap;
            selective_write_iterator<t_ve, t_out_f> m_Wit;
            bitmap_processing_state_t m_bm_ps_state;

            state_t(base_t p_Predicate, uint8_t * p_Out, bitmap_processing_state_t & p_bm_ps_state) :
                    m_Predicate(vectorlib::set1<t_ve, vector_base_t_granularity::value>(p_Predicate)),
                    m_Wit(p_Out),
                    m_bm_ps_state(p_bm_ps_state){}

            /** @brief To ensure that we eventually store a bitmap (e.g. processing only < 64 elements),
             *         this function needs to be called before calling the write_iterator's done() function.
             */
            MSV_CXX_ATTRIBUTE_FORCE_INLINE void done() {
                // if there is still a word > 0 or if word=0 with bitPos > 0 -> store the word
                if(m_bm_ps_state.m_active_word || m_bm_ps_state.m_bitPos) {
                    m_bitmap = vectorlib::set1<t_ve, vector_base_t_granularity::value>(
                            m_bm_ps_state.m_active_word);
                    m_Wit.write(m_bitmap, m_write_mask, 1);
                }
            }
        };

        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(vector_t p_Data, state_t & p_State) {
            // (1) Get result-mask from compare-operation
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
            vector_mask_t resultMask = t_compare<t_ve, vector_base_t_granularity::value>::apply(p_Data, p_State.m_Predicate);
#else
            vector_mask_t resultMask = t_compare::apply(p_Data, p_State.m_Predicate);
#endif
            // (2) Check if we are still in the current word boundary of base_t (e.g. uint64_t has 64 as boundary)
            if(!p_State.m_bm_ps_state.m_bitPos && !p_State.m_bm_ps_state.firstRun) {
                // Reached the boundary -> load word into bitmap vector, write vector to selective_write_iterator, update word and bitPos
                p_State.m_bitmap = vectorlib::set1<t_ve, vector_base_t_granularity::value>(
                        p_State.m_bm_ps_state.m_active_word);
                p_State.m_Wit.write(p_State.m_bitmap, p_State.m_write_mask, 1);

                p_State.m_bm_ps_state.m_active_word = 0;
                p_State.m_bm_ps_state.m_bitPos = 0;
            }

            // add resulting mask to current word: shift resulting bitmap by bitPos bits, then OR with current word
            // if resultMask == 0 we can skip it
            if(resultMask) {
                p_State.m_bm_ps_state.m_active_word |= (base_t) resultMask << p_State.m_bm_ps_state.m_bitPos;
            }

            // (3) Update bitPos index and firstRun flag
            p_State.m_bm_ps_state.m_bitPos = (p_State.m_bm_ps_state.m_bitPos + vector_element_count::value)
                                             % vector_base_t_granularity::value;
            if(p_State.m_bm_ps_state.firstRun) p_State.m_bm_ps_state.firstRun = false; // mark the first processing run
        }
    };


    template<
            template<class, int> class t_compare,
            class t_vector_extension,
            class t_out_IR_f,
            class t_in_data_f,
            typename std::enable_if_t<
                    // check if t_out_IR_f is a bitmap to instantiate the right operator according to its underlying IR
                    morphstore::is_bitmap_t< t_out_IR_f >::value
                    ,int> = 0
    >
    struct select_bm_wit_t {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

#ifndef COMPARE_OP_AS_TEMPLATE_CLASS
        using t_compare_special_ve = t_compare<t_ve, vector_base_t_granularity::value>;
        using t_compare_special_sc = t_compare<vectorlib::scalar<vectorlib::v64<uint64_t>>, vectorlib::scalar<vectorlib::v64<uint64_t>>::vector_helper_t::granularity::value>;
#endif

        static const column< t_out_IR_f > *
        apply(
                const column<t_in_data_f> * const inDataCol,
                const uint64_t val
        ) {
            using namespace vectorlib;

            const uint8_t * inData = inDataCol->get_data();
            const uint8_t * const initInData = inData;
            const size_t inDataCountLog = inDataCol->get_count_values();
            const size_t inDataSizeComprByte = inDataCol->get_size_compr_byte();
            const size_t inDataSizeUsedByte = inDataCol->get_size_used_byte();

            const size_t inCountLogCompr = inDataCol->get_count_values_compr();

            // number of columns needed to encode bitmaps -> pessimistic (all bits are set, and 1 word store's base_t elements)
            const size_t outBmCountEstimate = round_up_div(inDataCountLog, vector_base_t_granularity::value);

            auto outCol = new column< t_out_IR_f >(
                    get_size_max_byte_any_len< typename t_out_IR_f::t_inner_f >(outBmCountEstimate)
            );
            uint8_t * outPos = outCol->get_data();
            const uint8_t * const initOut = outPos;
            size_t outCountLog;
            size_t outSizeComprByte;

            // init bitmap processing state
            bitmap_processing_state_t bm_ps_state(0,0);

            // The state of the selective_write_iterator for the compressed output.
            typename select_processing_unit_wit<
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
                    t_ve, t_compare, typename t_out_IR_f::t_inner_f
#else
                    t_ve, t_compare_special_ve, typename t_out_IR_f::t_inner_f
#endif
            >::state_t witComprState(val, outPos, bm_ps_state);

            // Processing of the input column's compressed part using the specified
            // vector extension, compressed output.
            decompress_and_process_batch<
                    t_ve,
                    t_in_data_f,
                    select_processing_unit_wit,
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
                    t_compare,
#else
                    t_compare_special_ve,
#endif
                    typename t_out_IR_f::t_inner_f
            >::apply(
                    inData, inCountLogCompr, witComprState
            );

            if(inDataSizeComprByte == inDataSizeUsedByte) {
                // If the input column has no uncompressed rest part, we are done.

                // Eventually store bitmap-encoded word (e.g. if we process < 64 elements)
                witComprState.done();

                std::tie(
                        outSizeComprByte, std::ignore, outPos
                ) = witComprState.m_Wit.done();
                outCountLog = witComprState.m_Wit.get_count_values();
            }
            else {
                // If the input column has an uncompressed rest part.

                // Pad the input pointer such that it points to the beginning of
                // the input column's uncompressed rest part.
                inData = inDataCol->get_data_uncompr_start();
                // The size of the input column's uncompressed rest part.
                const size_t inSizeRestByte = initInData + inDataSizeUsedByte - inData;

                // Vectorized processing of the input column's uncompressed rest
                // part using the specified vector extension, compressed output.
                const size_t inDataSizeUncomprVecByte = round_down_to_multiple(
                        inSizeRestByte, vector_size_byte::value
                );
                decompress_and_process_batch<
                        t_ve,
                        uncompr_f,
                        select_processing_unit_wit,
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
                        t_compare,
#else
                        t_compare_special_ve,
#endif
                        typename t_out_IR_f::t_inner_f
                >::apply(
                        inData, convert_size<uint8_t, uint64_t>(inDataSizeUncomprVecByte), witComprState
                );

                // Finish the compressed output. This might already initialize the
                // output column's uncompressed rest part.
                uint8_t * outAppendUncompr;

                // The size of the input column's uncompressed rest that can only
                // be processed with scalar instructions.
                const size_t inSizeScalarRemainderByte = inSizeRestByte % vector_size_byte::value;

                // eventually store bitmap word if there is no scalar-remainder part,
                // otherwise we call state.done() in the remainder part (important: always call state.done() at the very end)
                if(!inSizeScalarRemainderByte) witComprState.done();

                std::tie(
                        outSizeComprByte, outAppendUncompr, outPos
                ) = witComprState.m_Wit.done();
                outCountLog = witComprState.m_Wit.get_count_values();

                if(inSizeScalarRemainderByte) {
                    // If there is such an uncompressed scalar rest.

                    // The state of the selective write_iterator for the
                    // uncompressed output.
                    typename select_processing_unit_wit<
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
                    scalar<v64<uint64_t>>, t_compare, uncompr_f
#else
                    scalar<v64<uint64_t>>, t_compare_special_sc, uncompr_f
                                                                 #endif
                                                                 >::state_t witUncomprState(
                            val,
                            outAppendUncompr,
                            witComprState.m_bm_ps_state // pass witComprState's prev. bm-state (otherwise starting from 0 again)
                    );

                    // Processing of the input column's uncompressed scalar rest
                    // part using scalar instructions, uncompressed output.
                    decompress_and_process_batch<
                    scalar<v64<uint64_t>>,
                            uncompr_f,
                            select_processing_unit_wit,
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
                            t_compare,
#else
                            t_compare_special_sc,
#endif
uncompr_f
>::apply(inData, convert_size<uint8_t, uint64_t>(inSizeScalarRemainderByte), witUncomprState);

                    // eventually store bitmap word
                    witUncomprState.done();

                    // Finish the output column's uncompressed rest part.
                    std::tie(
                            std::ignore, std::ignore, outPos
                    ) = witUncomprState.m_Wit.done();
                    outCountLog += witUncomprState.m_Wit.get_count_values();
                }
            }

            // Finish the output column.
            outCol->set_meta_data(
                    outCountLog, outPos - initOut, outSizeComprByte
            );

            return outCol;
        }
    };
}

#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_BM_COMPR_H