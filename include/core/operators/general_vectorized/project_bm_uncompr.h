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
 * @file project_bm_uncompr.h
 * @brief General vectorized project-operator using bitmaps as intermediates (instead of PL).
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_PROJECT_BM_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_PROJECT_BM_UNCOMPR_H

#include <core/utils/preprocessor.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/morphing/intermediates/transformations/transformation_algorithms.h>
#include <core/morphing/intermediates/bitmap.h>

namespace morphstore {

    template<class VectorExtension>
    struct project_bm_t_processing_unit {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static vector_t apply(
                base_t const * p_DataPtr,
                vector_t p_PosVector
        ) {
            return (vectorlib::gather<VectorExtension, vector_size_bit::value, sizeof(base_t)>(p_DataPtr, p_PosVector));
        }
    };

    template<class VectorExtension>
    struct project_bm_batch {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
                base_t const *& p_DataPtr,
                base_t const *& p_PosPtr,
                base_t *& p_OutPtr,
                size_t const p_Count
        ) {
            for(size_t i = 0; i < p_Count; ++i) {
                vector_t posVector = vectorlib::load<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(p_PosPtr);
                vector_t lookupVector =
                        project_bm_t_processing_unit<VectorExtension>::apply(
                                p_DataPtr,
                                posVector
                        );
                vectorlib::store<VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>(p_OutPtr, lookupVector);
                p_PosPtr += vector_element_count::value;
                p_OutPtr += vector_element_count::value;
            }
        }
    };

    template<class VectorExtension>
    struct project_bm_t {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static
                column<uncompr_f> const *
        apply(
                column< uncompr_f > const * const p_DataColumn,
                column< bitmap_f<uncompr_f> > const * const p_bitmapCol
        ) {
            base_t const * inDataPtr = p_DataColumn->get_data( );
            base_t const * inBmPtr = p_bitmapCol->get_data();
            size_t const inBmCount = p_bitmapCol->get_count_values();

            // intermediate column that holds the positions of one bitmap encoded word, used when iterating through bm-words;
            // one bitmap-word can hold up to vector_base_t_granularity::value-positions, e.g. uint64_t => 64 positions
            auto inPosCol = new column< position_list_f<> >(
                    vector_base_t_granularity::value * sizeof(base_t)
            );
            base_t * const initInPosPtr = inPosCol->get_data();

            // pessimistic allocation: every bit in bitmap is set (currently there is no metadata to get total popcount)
            auto outDataCol = new column<uncompr_f>(
                    inBmCount * vector_base_t_granularity::value * sizeof(base_t)
            );
            base_t * outDataPtr = outDataCol->get_data( );
            const base_t * const initOutDataPtr = outDataPtr;

            // iterate through input bitmap word-by-word, i.e. bitmap-encoded-64-bit words (word-granularity)
            for(size_t i = 0; i < inBmCount; ++i) {
                // check if bitmap encoded word is 0 -> if so, skip
                if(*inBmPtr) {
                    // re-interpreting casts, as the transform_IR-operator work with uint8_t pointers (like morph-operator)
                    const uint8_t * inBm8 = reinterpret_cast<const uint8_t *>(inBmPtr);
                    // starting point of intermediate pos.-list
                    uint8_t * inPos8 = reinterpret_cast<uint8_t *>(initInPosPtr);

                    // get positions of current bitmap encoded word
                    transform_IR_batch_t<VectorExtension, position_list_f<>, bitmap_f<> >::apply(
                            inBm8,
                            inPos8,
                            1, // we only process one word
                            i*vector_base_t_granularity::value
                    );

                    // re-cast to base_t alignment
                    base_t const * const inPosEnd = reinterpret_cast<base_t const * const>(inPos8);

                    // calculate the number of positions, from previous transformation
                    size_t const numberPositions = inPosEnd - initInPosPtr;

                    size_t const vectorCount = numberPositions / vector_element_count::value;
                    size_t const remainderCount = numberPositions % vector_element_count::value;

                    // tmp-pointer which is used in project_bm<> (to avoid that initInPosPtr gets increment or modified)
                    base_t const * inPosPtr_tmp = initInPosPtr;

                    // actual projection using position-list
                    project_bm_batch<VectorExtension>::apply(inDataPtr, inPosPtr_tmp, outDataPtr, vectorCount);
                    // we use only scalar processing here:
                    project_bm_batch<vectorlib::scalar<vectorlib::v64<uint64_t>>>::apply(inDataPtr, inPosPtr_tmp, outDataPtr, remainderCount);
                }
                // get next bitmap word
                ++inBmPtr;
            }

            size_t const outDataCount = outDataPtr - initOutDataPtr;
            outDataCol->set_meta_data(outDataCount, outDataCount * sizeof(base_t));

            return outDataCol;
        }
    };

    // @TODO: interfaces for simd-operators is still not harmonized
    template<
            class t_vector_extension,
            class t_out_data_f,
            class t_in_data_f,
            class t_in_IR_f,
            typename std::enable_if_t<
                    std::is_same< t_in_IR_f, bitmap_f<uncompr_f> >::value
            ,int> = 0
    >
    column<uncompr_f> const * project_bm(
            column< uncompr_f > const * const p_Data1Column,
            column< t_in_IR_f > const * const inBmCol
    ){
            return project_bm_t<t_vector_extension>::apply(p_Data1Column,inBmCol);
    }

}

#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_PROJECT_BM_UNCOMPR_H