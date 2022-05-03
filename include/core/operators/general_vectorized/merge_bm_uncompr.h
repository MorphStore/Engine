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
 * @file merge_bm_uncompr.h
 * @brief General vectorized merge-operator using bitmaps.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_MERGE_BM_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_MERGE_BM_UNCOMPR_H

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/morphing/intermediates/representation.h>

#include <core/utils/preprocessor.h>
//#include <core/operators/interfaces/merge_bm_pl.h>

#include <cstring>

namespace morphstore {

    template<class VectorExtension>
    struct merge_bm_batch {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
                base_t const *& p_BmPtrSmaller,
                base_t const *& p_BmPtrLarger,
                base_t *& p_OutPtr,
                size_t const p_Count
        ) {
            for(size_t i = 0; i < p_Count; ++i) {
                // load to register
                vector_t bmSmallerVector = vectorlib::load<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(p_BmPtrSmaller);
                vector_t bmLargerVector = vectorlib::load<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(p_BmPtrLarger);

                // execute bitwise logical OR-operation
                vector_t mergeVector = bitwise_or<VectorExtension>(bmSmallerVector, bmLargerVector);

                // store and increment
                vectorlib::store<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(p_OutPtr, mergeVector);
                p_BmPtrSmaller += vector_element_count::value;
                p_BmPtrLarger += vector_element_count::value;
                p_OutPtr += vector_element_count::value;
            }
        }
    };

    template<class VectorExtension>
    struct merge_bm_t {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static
        column< bitmap_f<uncompr_f> > const *
        apply(
                column< bitmap_f<uncompr_f> > const * const p_Data1Column,
                column< bitmap_f<uncompr_f> > const * const p_Data2Column,
                const size_t p_OutPosCountEstimate = 0
        ) {
            // differentiate the input bitmaps according to their sizes
            const column< bitmap_f<uncompr_f> > * smallerBmCol;
            const column< bitmap_f<uncompr_f> > * largerBmCol;
            if(p_Data1Column->get_size_used_byte() < p_Data2Column->get_size_used_byte()) {
                smallerBmCol = p_Data1Column;
                largerBmCol = p_Data2Column;
            } else {
                smallerBmCol = p_Data2Column;
                largerBmCol = p_Data1Column;
            }

            const base_t * smallerBmPtr = smallerBmCol->get_data();
            const base_t * largerBmPtr = largerBmCol->get_data();
            //const base_t * const endInBmSmaller = smallerBmPtr + smallerBmCol->get_count_values();
            const base_t * const endInBmLarger = largerBmPtr + largerBmCol->get_count_values();

            // if no estimation is given, use the larger input bitmap size for allocation
            auto outBmCol = new column< bitmap_f<uncompr_f> >(
                    bool(p_OutPosCountEstimate)
                    // use given estimate
                    ? (p_OutPosCountEstimate * sizeof(base_t))
                    // use pessimistic estimate
                    :
                    largerBmCol->get_size_used_byte()
            );

            base_t * outBmPtr = outBmCol->get_data( );
            base_t * const outBmPtrOrigin = const_cast< base_t * const >(outBmPtr);

            // we process the smaller part first
            size_t const vectorCount = smallerBmCol->get_count_values() / vector_element_count::value;
            size_t const remainderCount = smallerBmCol->get_count_values() % vector_element_count::value;

            merge_bm_batch<VectorExtension>::apply(smallerBmPtr, largerBmPtr, outBmPtr, vectorCount);

            merge_bm_batch<scalar<v64<uint64_t>>>::apply(smallerBmPtr, largerBmPtr, outBmPtr, remainderCount);

            // memcpy() remaining bits from larger bitmap to output bitmap (if there is any count left)
            if(largerBmPtr < endInBmLarger){
                const size_t remainingCount = endInBmLarger - largerBmPtr;
                std::memcpy(outBmPtr, largerBmPtr, remainingCount * sizeof(base_t));
                outBmPtr += remainingCount;
            }

            size_t const outDataCount = outBmPtr - outBmPtrOrigin;
            outBmCol->set_meta_data(outDataCount, outDataCount * sizeof(base_t) );

            return outBmCol;
        }
    };

    template<
            class VectorExtension,
            class t_out_IR_f,
            class t_in_IR_l_f,
            class t_in_IR_r_f,
            typename std::enable_if_t<
                    // enable only if all input formats are uncompressed bitmaps
                    std::is_same< t_in_IR_l_f, bitmap_f<uncompr_f> >::value &&
                    std::is_same< t_in_IR_r_f, bitmap_f<uncompr_f> >::value &&
                    std::is_same< t_out_IR_f,  bitmap_f<uncompr_f> >::value
            ,int> = 0
    >
    column< bitmap_f<uncompr_f>  > const *
    merge_sorted(
            column< bitmap_f<uncompr_f>  > const * const p_Data1Column,
            column< bitmap_f<uncompr_f>  > const * const p_Data2Column,
            const size_t p_OutPosCountEstimate = 0
    ) {
        return merge_bm_t<VectorExtension>::apply(p_Data1Column,p_Data2Column,p_OutPosCountEstimate);
    }
}

#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_MERGE_BM_UNCOMPR_H
