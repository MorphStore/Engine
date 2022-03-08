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
 * @file select_bm_uncompr.h
 * @brief General vectorized select-operator using bitmaps as intermediates (instead of PL).
 */

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_BM_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_BM_UNCOMPR_H

#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/morphing/intermediates/bitmap.h>

namespace morphstore {

    using namespace vectorlib;
    template<class VectorExtension,  template< class, int > class Operator>
    struct select_processing_unit {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
                vector_mask_t
        apply(
                vector_t const p_DataVector,
        vector_t const p_PredicateVector
        ) {
            return Operator<VectorExtension,VectorExtension::vector_helper_t::granularity::value>::apply(
                    p_DataVector,
                    p_PredicateVector
            );
        }

        /** @brief To ensure that we eventually store a bitmap (e.g. processing only < 64 elements),
         *         this function needs to called before the operator has finished its processing.
         */
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void done(uint64_t *& p_OutPtr, bitmap_processing_state_t & p_bm_state) {
            // if there is still a word > 0 or if word=0 with bitPos > 0 -> store the word
            if(p_bm_state.m_active_word || p_bm_state.m_bitPos) {
                *p_OutPtr = p_bm_state.m_active_word;
                p_OutPtr++;
            }
        }
    };

    template<class VectorExtension,  template< class, int > class Operator>
    struct select_bm_batch {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
                base_t const *& p_DataPtr,
                base_t const p_Predicate,
                base_t *& p_OutPtr,
                size_t const p_Count,
                bitmap_processing_state_t & p_bm_ps_state
        ) {
            vector_t const predicateVector = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(
                    p_Predicate);
            // init local values from global bm-state
            base_t bitPos = p_bm_ps_state.m_bitPos;
            base_t word = p_bm_ps_state.m_active_word;

            for (size_t i = 0; i < p_Count; ++i) {

                vector_t dataVector = vectorlib::load<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(
                        p_DataPtr);
                vector_mask_t resultMask =
                        select_processing_unit<VectorExtension, Operator>::apply(
                                dataVector,
                                predicateVector
                        );

                // if we reach the boundary of base_t => store word in column, increment pointer & reset word
                if(!p_bm_ps_state.m_bitPos && !p_bm_ps_state.firstRun) {
                    *p_OutPtr = p_bm_ps_state.m_active_word;
                    p_OutPtr++;
                    p_bm_ps_state.m_active_word = 0;
                }

                // add resulting mask to current word:
                // shift resulting bitmap by bitPos bits, then OR with current word
                // if resultMask == 0 we can skip it
                if(resultMask){
                    p_bm_ps_state.m_active_word |= (base_t) resultMask << p_bm_ps_state.m_bitPos;
                }

                p_bm_ps_state.m_bitPos = (p_bm_ps_state.m_bitPos + vector_element_count::value) % vector_base_t_granularity::value;
                p_DataPtr += vector_element_count::value;
                if(p_bm_ps_state.firstRun) p_bm_ps_state.firstRun = false; // mark the first processing run
            }
        }
    };

    template<class VectorExtension, template< class, int > class Operator>
    struct select_bm_t {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static
                column< bitmap_f<uncompr_f> > const *
        apply(
                column< uncompr_f > const * const p_DataColumn,
        base_t const p_Predicate
        ) {

            size_t const inDataCount = p_DataColumn->get_count_values();
            base_t const * inDataPtr = p_DataColumn->get_data( );

            // number of columns needed to encode bitmaps -> pessimistic
            const size_t outBmCountEstimate = round_up_div(inDataCount, vector_base_t_granularity::value);

            auto outDataCol = new column< bitmap_f<uncompr_f> >(
                    outBmCountEstimate * sizeof(base_t)
            );

            base_t * outDataPtr = outDataCol->get_data();
            base_t * const outDataPtrOrigin = const_cast< base_t * const >(outDataPtr);

            size_t const vectorCount = inDataCount / vector_element_count::value;
            size_t const remainderCount = inDataCount % vector_element_count::value;

            // init bitmap processing GLOBAL state (from bitmap.h)
            bitmap_processing_state_t bm_ps_state(0,0);

            select_bm_batch<VectorExtension, Operator>::apply(inDataPtr, p_Predicate, outDataPtr, vectorCount, bm_ps_state);

            select_bm_batch<scalar<v64<uint64_t>>, Operator>::apply(inDataPtr, p_Predicate, outDataPtr, remainderCount, bm_ps_state);

            // Eventually store bitmap-encoded word (e.g. if we process < 64 elements)
            select_processing_unit<VectorExtension, Operator>::done(outDataPtr, bm_ps_state);

            size_t const outDataCount = outDataPtr - outDataPtrOrigin;

            outDataCol->set_meta_data(outDataCount, outDataCount*sizeof(base_t));

            return outDataCol;
        }
    };


    // @TODO: interfaces for simd-operators is still not harmonized
    template<
            template< class, int > class Operator,
            class t_vector_extension,
            class t_out_IR_f,
            class t_in_data_f,
            typename std::enable_if_t<
                    // check if t_out_IR_f is an uncompressed bitmap to instantiate the right operator according to its underlying IR
                    std::is_same< t_out_IR_f, bitmap_f<uncompr_f> >::value
    ,int> = 0
    >
    column<t_out_IR_f> const *
    select(
            column< uncompr_f > const * const p_DataColumn,
            typename t_vector_extension::vector_helper_t::base_t const p_Predicate
    ){
            return select_bm_t<t_vector_extension, Operator>::apply(p_DataColumn, p_Predicate);
    }

}

#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_BM_UNCOMPR_H