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
#include <core/memory/management/utils/alignment_helper.h>
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

#include <tuple>

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



// @todo It would be nice if the core-operator's template argument for the
// comparison operation/primitive could be a template-class. However, the
// compiler does not like this. Maybe I'm doing something wrong...
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

    struct state_t {
        const vector_t m_Predicate;
        vector_t m_Pos;
        // @todo This can be static.
        const vector_t m_Inc;
        write_iterator<t_ve, t_out_f> m_Wit;

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
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
        vector_mask_t mask = t_compare<t_ve, vector_base_t_granularity::value>::apply(p_Data, p_State.m_Predicate);
#else
        vector_mask_t mask = t_compare::apply(p_Data, p_State.m_Predicate);
#endif
        if(mask)
            p_State.m_Wit.write(p_State.m_Pos, mask);
        p_State.m_Pos = vector::add<t_ve>::apply(p_State.m_Pos, p_State.m_Inc);
    }
};
    
template<
        template<class, int> class t_compare,
        class t_vector_extension,
        class t_out_pos_f,
        class t_in_data_f
>
// @todo We cannot call it select or select_t at the moment, because it has
// other requirements for t_op than the select-struct in the general interface.
struct my_select_wit_t {
    using t_ve = t_vector_extension;
    IMPORT_VECTOR_BOILER_PLATE(t_ve)
#ifndef COMPARE_OP_AS_TEMPLATE_CLASS
    using t_compare_special_ve = t_compare<t_ve, vector_base_t_granularity::value>;
    using t_compare_special_sc = t_compare<vector::scalar<vector::v64<uint64_t>>, vector::scalar<vector::v64<uint64_t>>::vector_helper_t::granularity::value>;
#endif
    
    static const column<t_out_pos_f> * apply(
            const column<t_in_data_f> * const inDataCol,
            const uint64_t val,
            const size_t outPosCountEstimate = 0
    ) {
        using namespace vector;
        
        const uint8_t * inData = inDataCol->get_data();
        const uint8_t * const initInData = inData;
        const size_t inDataSizeComprByte = inDataCol->get_size_compr_byte();
        const size_t inDataSizeUsedByte = inDataCol->get_size_used_byte();

        // If no estimate is provided: Pessimistic allocation size (for
        // uncompressed data), reached only if all input data elements pass the
        // selection.
        // @todo Due to the input column's uncompressed rest part, we might
        // actually need more space, if the input column is the product of some
        // other query operator.
        auto outPosCol = new column<t_out_pos_f>(
                bool(outPosCountEstimate)
                // use given estimate
                ? get_size_max_byte_any_len<t_out_pos_f>(outPosCountEstimate)
                // use pessimistic estimate
                : get_size_max_byte_any_len<t_out_pos_f>(inDataCol->get_count_values())
        );
        uint8_t * outPos = outPosCol->get_data();
        const uint8_t * const initOutPos = outPos;
        size_t outCountLog;
        size_t outSizeComprByte;

        // The state of the write_iterator for the compressed output.
        typename select_processing_unit_wit<
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
                t_ve, t_compare, t_out_pos_f
#else
                t_ve, t_compare_special_ve, t_out_pos_f
#endif
        >::state_t witComprState(val, outPos);
        
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
                t_out_pos_f
        >::apply(
                inData, inDataSizeComprByte, witComprState
        );
        
        if(inDataSizeComprByte == inDataSizeUsedByte) {
            // If the input column has no uncompressed rest part, we are done.
            std::tie(
                    outSizeComprByte, std::ignore, outPos
            ) = witComprState.m_Wit.done();
            outCountLog = witComprState.m_Wit.get_count_values();
        }
        else {
            // If the input column has an uncompressed rest part.
            
            // Pad the input pointer such that it points to the beginning of
            // the input column's uncompressed rest part.
            inData = create_aligned_ptr(inData);
            // The size of the input column's uncompressed rest part.
            const size_t inSizeRestByte = initInData + inDataSizeUsedByte - inData;
            
            // Vectorized processing of the input column's uncompressed rest
            // part using the specified vector extension, compressed output.
            decompress_and_process_batch<
                    t_ve,
                    uncompr_f,
                    select_processing_unit_wit,
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
                    t_compare,
#else
                    t_compare_special_ve,
#endif
                    t_out_pos_f
            >::apply(
                    inData,
                    inSizeRestByte / vector_size_byte::value,
                    witComprState
            );
            
            // Finish the compressed output. This might already initialize the
            // output column's uncompressed rest part.
            bool outAddedPadding;
            std::tie(
                    outSizeComprByte, outAddedPadding, outPos
            ) = witComprState.m_Wit.done();
            outCountLog = witComprState.m_Wit.get_count_values();
            
            // The size of the input column's uncompressed rest that can only
            // be processed with scalar instructions.
            const size_t inSizeScalarRemainderByte = inSizeRestByte % vector_size_byte::value;
            if(inSizeScalarRemainderByte) {
                // If there is such an uncompressed scalar rest.
                
                // Pad the output pointer such that it points to the beginning
                // of the output column's uncompressed rest, if this has not
                // already been done when finishing the output column's
                // compressed part.
                if(!outAddedPadding)
                    outPos = create_aligned_ptr(outPos);
                
                // The state of the write_iterator for the uncompressed output.
                typename select_processing_unit_wit<
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
                        scalar<v64<uint64_t>>, t_compare, uncompr_f
#else
                        scalar<v64<uint64_t>>, t_compare_special_sc, uncompr_f
#endif
                >::state_t witUncomprState(val, outPos);

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
                >::apply(
                        inData, inSizeScalarRemainderByte, witUncomprState
                );
                
                // Finish the output column's uncompressed rest part.
                std::tie(
                        std::ignore, std::ignore, outPos
                ) = witUncomprState.m_Wit.done();
                outCountLog += witUncomprState.m_Wit.get_count_values();
            }
        }

        // Finish the output column.
        outPosCol->set_meta_data(
                outCountLog, outSizeComprByte, outPos - initOutPos
        );

        return outPosCol;
    }
};

#undef COMPARE_OP_AS_TEMPLATE_CLASS

}
#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_SELECT_COMPR_H
