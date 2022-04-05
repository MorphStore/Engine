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
#include <core/morphing/format.h>
#include <core/morphing/write_iterator_IR.h>
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
            class t_out_f,
            class t_IR_dst_f,
            class t_IR_src_f
    >
    struct select_processing_unit_wit {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

        struct state_t {
            const vector_t m_Predicate;
            const vector_mask_t m_write_mask = 1; // we just want to store the first word in the vector
            base_t m_bitmap;
            // selective write-iterator including IR-transformation
            selective_write_iterator_IR<t_ve, t_out_f, t_IR_dst_f, t_IR_src_f> m_Wit;
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
                    m_bitmap = m_bm_ps_state.m_active_word;

                    m_Wit.write_bitmap_word(m_bitmap);
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
            // (2) add resulting mask to current word: shift resulting bitmap by bitPos bits, then OR with current word
            // if resultMask == 0 we can skip it
            if(resultMask) {
                p_State.m_bm_ps_state.m_active_word |= (base_t) resultMask << p_State.m_bm_ps_state.m_bitPos;
            }

            // (3) Update bitPos index and firstRun flag
            p_State.m_bm_ps_state.m_bitPos = (p_State.m_bm_ps_state.m_bitPos + vector_element_count::value)
                                             % vector_base_t_granularity::value;

            // (4) Check if we are still in the current word boundary of base_t (e.g. uint64_t has 64 as boundary)
            if(!p_State.m_bm_ps_state.m_bitPos) {
                // Reached the boundary -> load word into bitmap vector, write vector to selective_write_iterator, update word and bitPos
                p_State.m_bitmap = p_State.m_bm_ps_state.m_active_word;
                p_State.m_Wit.write_bitmap_word(p_State.m_bitmap);

                p_State.m_bm_ps_state.m_active_word = 0;
                p_State.m_bm_ps_state.m_bitPos = 0;
            }

            // update internal state of write-iterator for IR-transformation
            p_State.m_Wit.update();
        }
    };


    template<
            template<class, int> class t_compare,
            class t_vector_extension,
            class t_IR_dst_f,
            class t_in_data_f,
            typename std::enable_if_t<
                    // check if t_IR_dst_f is an IR-type to enable this
                    morphstore::is_intermediate_representation_t< t_IR_dst_f >::value
            ,int> = 0
    >
    struct select_bm_wit_t {
        // write iterators currently work only on uncompressed IRs -> fetch the underlying data structure
        using t_IR_dest_uncompr =
                typename std::conditional<
                        morphstore::is_bitmap_t<t_IR_dst_f>::value,
                        bitmap_f<>,
                        position_list_f<>
                >::type;
        using t_IR_src_uncompr = bitmap_f<>;
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

#ifndef COMPARE_OP_AS_TEMPLATE_CLASS
        using t_compare_special_ve = t_compare<t_ve, vector_base_t_granularity::value>;
        using t_compare_special_sc = t_compare<vectorlib::scalar<vectorlib::v64<uint64_t>>, vectorlib::scalar<vectorlib::v64<uint64_t>>::vector_helper_t::granularity::value>;
#endif

        static const column< t_IR_dst_f > *
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

            // if output IR-type is a position-list, we need another allocation strategy than for bitmaps
            const size_t outCount =
                    (morphstore::is_bitmap_t<t_IR_dst_f>::value)
                    ?
                    round_up_div(inDataCountLog, vector_base_t_granularity::value) // pessimistic (all bits are set, and 1 word store's base_t elements)
                    :
                    inDataCountLog; // pessimistic

            auto outCol = new column< t_IR_dst_f >(
                    get_size_max_byte_any_len< typename t_IR_dst_f::t_inner_f >(outCount)
            );
            uint8_t * outPos = outCol->get_data();
            //const uint8_t * const initOut = outPos;
            size_t outCountLog;
            size_t outSizeComprByte;

            // init bitmap processing state
            bitmap_processing_state_t bm_ps_state(0,0);

            // The state of the selective_write_iterator for the compressed output.
            typename select_processing_unit_wit<
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
                    t_ve, t_compare, typename t_IR_dst_f::t_inner_f,
                    t_IR_dest_uncompr, t_IR_src_uncompr
#else
                    t_ve, t_compare_special_ve, typename t_IR_dst_f::t_inner_f,
                    t_IR_dest_uncompr, t_IR_src_uncompr
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
                    typename t_IR_dst_f::t_inner_f,
                    t_IR_dest_uncompr,
                    t_IR_src_uncompr
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
                        typename t_IR_dst_f::t_inner_f,
                        t_IR_dest_uncompr,
                        t_IR_src_uncompr
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
                    scalar<v64<uint64_t>>, t_compare, uncompr_f,
                    t_IR_dest_uncompr, t_IR_src_uncompr
#else
                    scalar<v64<uint64_t>>, t_compare_special_sc, uncompr_f,
                    t_IR_dest_uncompr, t_IR_src_uncompr
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
                            uncompr_f,
                            t_IR_dest_uncompr,
                            t_IR_src_uncompr
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
                    outCountLog, /*utPos - initOut,*/ outSizeComprByte
            );

            return outCol;
        }
    };
}

#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_BM_COMPR_H