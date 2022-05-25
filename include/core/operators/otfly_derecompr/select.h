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
 * @file select.h
 * @brief Select-operator based on the vector-lib, weaving the operator's core
 * into the decompression routine of the input data's format.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_OTFLY_DERECOMPR_SELECT_H
#define MORPHSTORE_CORE_OPERATORS_OTFLY_DERECOMPR_SELECT_H

// @todo Include this as soon as the interfaces are harmonized.

#include <core/utils/preprocessor.h>
#include <core/storage/column.h>
#include <core/morphing/format.h>
#include <core/morphing/write_iterator.h>
#include <core/operators/interfaces/select.h>
#include <core/memory/management/utils/alignment_helper.h>

#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>


#include <tuple>

#include <cstdint>

namespace morphstore {

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
        selective_write_iterator<t_ve, t_out_f> m_Wit;

        state_t(base_t p_Predicate, uint8_t * p_Out, base_t p_Pos) :
                m_Predicate(vectorlib::set1<t_ve, vector_base_t_granularity::value>(p_Predicate)),
                m_Pos(vectorlib::set_sequence<t_ve, vector_base_t_granularity::value>(p_Pos, 1)),
                m_Inc(vectorlib::set1<t_ve, vector_base_t_granularity::value>(vector_element_count::value)),
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
        p_State.m_Pos = vectorlib::add<t_ve>::apply(p_State.m_Pos, p_State.m_Inc);
    }
};

template<class t_vector_extension, 
          template<class, int> class t_compare, 
          class t_out_pos_f, 
          class t_in_data_f
          >

struct select_t
{
    using t_ve = t_vector_extension;
    IMPORT_VECTOR_BOILER_PLATE(t_ve)
    #ifndef COMPARE_OP_AS_TEMPLATE_CLASS
    using t_compare_special_ve = t_compare<t_ve, vector_base_t_granularity::value>;
    using t_compare_special_sc = t_compare<vectorlib::scalar<vectorlib::v64<uint64_t>>, vectorlib::scalar<vectorlib::v64<uint64_t>>::vector_helper_t::granularity::value>;
    #endif

    static
    const column<t_out_pos_f> * apply(
    //static const column<t_out_pos_f> * apply(
            const column<t_in_data_f> * const inDataCol,
            const uint64_t val,
            const size_t outPosCountEstimate = 0
    ) {
    
    

        using namespace vectorlib;
        
        const uint8_t * inData = inDataCol->get_data();
        const uint8_t * const initInData = inData;
        const size_t inDataCountLog = inDataCol->get_count_values();
        const size_t inDataSizeComprByte = inDataCol->get_size_compr_byte();
        const size_t inDataSizeUsedByte = inDataCol->get_size_used_byte();
        
        const size_t inCountLogCompr = inDataCol->get_count_values_compr();

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
                : get_size_max_byte_any_len<t_out_pos_f>(inDataCountLog)
        );
        uint8_t * outPos = outPosCol->get_data();
        const uint8_t * const initOutPos = outPos;
        size_t outCountLog;
        size_t outSizeComprByte;

        // The state of the selective_write_iterator for the compressed output.
        typename select_processing_unit_wit<
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
                t_ve, t_compare, t_out_pos_f
#else
                t_ve, t_compare_special_ve, t_out_pos_f
#endif
        >::state_t witComprState(val, outPos, 0);
        
        // Processing of the input column's compressed part using the specified
        // vector extension, compressed output.
        decompress_and_process_batch<
                t_vector_extension,
                t_in_data_f,
                select_processing_unit_wit,
#ifdef COMPARE_OP_AS_TEMPLATE_CLASS
                t_compare,
#else
                 t_compare<t_vector_extension, vector_base_t_granularity::value>,
#endif
                t_out_pos_f
        >::apply(
                inData, inCountLogCompr, witComprState
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
                    t_out_pos_f
            >::apply(
                    inData, convert_size<uint8_t, uint64_t>(inDataSizeUncomprVecByte), witComprState
            );
            
            // Finish the compressed output. This might already initialize the
            // output column's uncompressed rest part.
            uint8_t * outAppendUncompr;
            std::tie(
                    outSizeComprByte, outAppendUncompr, outPos
            ) = witComprState.m_Wit.done();
            outCountLog = witComprState.m_Wit.get_count_values();
            
            // The size of the input column's uncompressed rest that can only
            // be processed with scalar instructions.
            const size_t inSizeScalarRemainderByte = inSizeRestByte % vector_size_byte::value;
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
                        inCountLogCompr + inDataSizeUncomprVecByte / sizeof(base_t)
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
                >::apply(
                        inData, convert_size<uint8_t, uint64_t>(inSizeScalarRemainderByte), witUncomprState
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
                outCountLog, outPos - initOutPos, outSizeComprByte
        );

        return outPosCol;
    }
};

#undef COMPARE_OP_AS_TEMPLATE_CLASS

}
#endif //MORPHSTORE_CORE_OPERATORS_OTFLY_DERECOMPR_SELECT_H
