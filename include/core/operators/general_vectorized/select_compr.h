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
 * @file select_compr.h
 * @brief Select-operator based on the vector-lib, weaving the operator's core
 * into the decompression routine of the input data's format.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_COMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_COMPR_H

// @todo Include this as soon as the interfaces are harmonized.
//#include <core/operators/interfaces/select.h>
// @todo Include this for the operator on the processing-unit-level.
//#include <core/operators/general_vectorized/select_uncompr.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/general_vector.h>
#include <vector/primitives/calc.h>
#include <vector/primitives/create.h>
#include <vector/primitives/io.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/scalar/primitives/calc_scalar.h>
#include <vector/scalar/primitives/create_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>

#include <cstdint>

namespace morphstore {

#if 1
// Hand-written scalar implementation, fast.
template<template<class, int> class t_op>
struct select_processing_unit {
    // We need a nested struct, because
    // (1) we must be able to hand a class with a single template argument (for
    //     the vector extension) to decompress_and_process
    // (2) we still want to be generic w.r.t. the comparison operator
    template<class t_vector_extension>
    struct type {
        struct state_t {
            const uint64_t m_Predicate;
            uint64_t * m_Out;
            size_t m_Pos;

            state_t(uint64_t p_Predicate, uint64_t * p_Out) :
            m_Predicate(p_Predicate), m_Out(p_Out), m_Pos(0) {
                //
            }
        };

        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(uint64_t p_Data, state_t & p_State) {
            if(p_Data == p_State.m_Predicate)
                *(p_State.m_Out)++ = p_State.m_Pos;
            p_State.m_Pos++;
        }
    };
};
#else
// VectorLib implementation, quite slow for scalar (maybe because of popcount?).
template<template<class, int> class t_op>
struct select_processing_unit {
    // We need a nested struct, because
    // (1) we must be able to hand a class with a single template argument (for
    //     the vector extension) to decompress_and_process
    // (2) we still want to be generic w.r.t. the comparison operator
    template<class t_vector_extension>
    struct type {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
        
        struct state_t {
            const vector_t m_PredVec;
            base_t * m_Out;
            vector_t m_PosVec;
            const vector_t m_AddVec;

            state_t(base_t p_PredScalar, base_t * p_Out) :
                    m_PredVec(vector::set1<t_vector_extension, vector_base_t_granularity::value>(p_PredScalar)),
                    m_Out(p_Out),
                    m_PosVec(vector::set_sequence<t_vector_extension, vector_base_t_granularity::value>(0, 1)),
                    m_AddVec(vector::set1<t_vector_extension, vector_base_t_granularity::value>(vector_element_count::value))
            {
                //
            }
        };

        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(vector_t p_Data, state_t & p_State) {
            vector_mask_t resultMask = t_op<t_vector_extension, vector_base_t_granularity::value>::apply(
                    p_Data, p_State.m_PredVec
            );
            vector::compressstore<t_vector_extension, vector::iov::UNALIGNED, vector_size_bit::value>(
                    p_State.m_Out, p_State.m_PosVec, resultMask
            );
            p_State.m_PosVec = vector::add<t_vector_extension, vector_base_t_granularity::value>::apply(
                    p_State.m_PosVec, p_State.m_AddVec
            );
            p_State.m_Out += __builtin_popcount(resultMask);
        }
    };
};
#endif
    
template<
        template<class, int> class t_op,
        class t_vector_extension,
        class t_in_data_f
>
// @todo We cannot call it select or select_t at the moment, because it has
// other requirements for t_op than the select-struct in the general interface.
struct my_select_t {
    static const column<uncompr_f> * apply(
            const column<t_in_data_f> * const inDataCol,
            const uint64_t val,
            const size_t outPosCountEstimate = 0
    ) {
        const uint8_t * inData = inDataCol->get_data();

        // If no estimate is provided: Pessimistic allocation size (for
        // uncompressed data), reached only if all input data elements pass the
        // selection.
        auto outPosCol = new column<uncompr_f>(
                bool(outPosCountEstimate)
                // use given estimate
                ? (outPosCountEstimate * sizeof(uint64_t))
                // use pessimistic estimate
                : uncompr_f::get_size_max_byte(inDataCol->get_count_values())
        );
        
        uint64_t * outPos = outPosCol->get_data();
        const uint64_t * const initOutPos = outPos;

        typename select_processing_unit<t_op>::template type<t_vector_extension>::state_t s(val, outPos);
        decompress_and_process_batch<
                t_vector_extension, t_in_data_f, select_processing_unit<t_op>::template type
        >::apply(
                inData, inDataCol->get_size_used_byte(), s
        );

        const size_t outPosCount = s.m_Out - initOutPos;
        outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));

        return outPosCol;
    }
};



// Hand-written scalar implementation, fast.
template<template<class, int> class t_op, class t_wit>
struct select_processing_unit_wit {
    // We need a nested struct, because
    // (1) we must be able to hand a class with a single template argument (for
    //     the vector extension) to decompress_and_process
    // (2) we still want to be generic w.r.t. the comparison operator
    template<class t_vector_extension>
    struct type {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
        
        struct state_t {
            const vector_t m_Predicate;
            vector_t m_Pos;
            // @todo This can be static.
            const vector_t m_Inc;
            t_wit m_Wit;

            state_t(base_t p_Predicate, uint8_t * p_Out) :
                    m_Predicate(vector::set1<t_ve, vector_base_t_granularity::value>(p_Predicate)),
                    m_Pos(vector::set_sequence<t_ve, vector_base_t_granularity::value>(0, 1)),
                    m_Inc(vector::set1<t_ve, vector_base_t_granularity::value>(vector_element_count::value)),
                    m_Wit(p_Out)
            {
                //
            }
        };

        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(vector_t p_Data, state_t & p_State) {
            vector_mask_t mask = vector::equal<t_ve>::apply(p_Data, p_State.m_Predicate);
            if(mask)
                p_State.m_Wit.write(p_State.m_Pos, mask);
            p_State.m_Pos = vector::add<t_ve>::apply(p_State.m_Pos, p_State.m_Inc);
        }
    };
};
    
template<
        template<class, int> class t_op,
        class t_vector_extension,
        class t_out_pos_f,
        class t_in_data_f
>
// @todo We cannot call it select or select_t at the moment, because it has
// other requirements for t_op than the select-struct in the general interface.
struct my_select_wit_t {
    IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
    
    static const column<t_out_pos_f> * apply(
            const column<t_in_data_f> * const inDataCol,
            const uint64_t val,
            const size_t outPosCountEstimate = 0
    ) {
        const uint8_t * inData = inDataCol->get_data();

        // If no estimate is provided: Pessimistic allocation size (for
        // uncompressed data), reached only if all input data elements pass the
        // selection.
        auto outPosCol = new column<t_out_pos_f>(
                bool(outPosCountEstimate)
                // use given estimate
                ? (outPosCountEstimate * sizeof(uint64_t))
                // use pessimistic estimate
                : t_out_pos_f::get_size_max_byte(inDataCol->get_count_values())
        );
        
        uint8_t * outPos = outPosCol->get_data();

        typename select_processing_unit_wit<t_op, write_iterator<t_vector_extension, t_out_pos_f>>::template type<t_vector_extension>::state_t s(val, outPos);
        decompress_and_process_batch<
                t_vector_extension, t_in_data_f, select_processing_unit_wit<t_op, write_iterator<t_vector_extension, t_out_pos_f>>::template type
        >::apply(
                inData, inDataCol->get_size_used_byte(), s
        );

        const size_t outPosCount = s.m_Wit.get_count();
        s.m_Wit.done();
        outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));

        return outPosCol;
    }
};

}
#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_COMPR_H
