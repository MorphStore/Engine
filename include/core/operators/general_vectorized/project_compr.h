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
 * If not, see <http://www.gnu.orga/licenses/>.                                                *
 **********************************************************************************************/

/**
 * @file project_compr.h
 * @brief Project-operator based on the vector-lib, weaving the operator's core
 * into the decompression routine of the input data's format.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_PROJECT_COMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_PROJECT_COMPR_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/write_iterator.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <iostream>
#include <tuple>
#include <type_traits>

#include <cstdint>
#include <cstring>

//
#include <core/morphing/default_formats.h>
#include <core/storage/replicated_column.h>
//#include <core/operators/interfaces/project.h>

namespace morphstore {

template<
        class t_vector_extension, class t_out_data_f, class t_in_data_f
>
struct project_processing_unit_wit {
    using t_ve = t_vector_extension;
    IMPORT_VECTOR_BOILER_PLATE(t_ve)

    struct state_t {
        nonselective_write_iterator<t_ve, t_out_data_f> m_Wit;
        typename random_read_access<t_ve, t_in_data_f>::type m_Rra;

        state_t(uint8_t * p_OutData, const base_t * p_InData) :
                m_Wit(p_OutData), m_Rra(p_InData)
        {
            //
        }
    };

    MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
            vector_t p_Data, state_t & p_State
    ) {
        p_State.m_Wit.write(p_State.m_Rra.get(p_Data));
    }
};
    
template<
        class t_vector_extension,
        class t_out_data_f,
        class t_in_data_f,
        class t_in_pos_f
>
class my_project_wit_t {
    using t_ve = t_vector_extension;
    IMPORT_VECTOR_BOILER_PLATE(t_ve)
    
public:
    static const column<t_out_data_f> * apply(
            const column<t_in_data_f> * const inDataCol,
            const column<t_in_pos_f> * const inPosCol
    ) {
        using namespace vectorlib;
        
        //if(inDataCol->template prepare_for_random_access<t_ve>())
            // @todo It would be nice to use warn() from the logger, but this
            // always outputs to cout, which interferes with our checks of the
            // program's output.
            //std::cerr
            //        << "[warn]: a project-operator's input data column was "
            //           "prepared for random access within the "
            //           "project-operator, which might corrupt measurements of "
            //           "the operator's execution time"
            //        << std::endl;
        
        const base_t * inData = inDataCol->get_data();
        
        const uint8_t * inPos = inPosCol->get_data();
        const uint8_t * const initInPos = inPos;
        const size_t inPosCountLog = inPosCol->get_count_values();
        const size_t inPosSizeComprByte = inPosCol->get_size_compr_byte();
        const size_t inPosSizeUsedByte = inPosCol->get_size_used_byte();
        
        // @todo Simplify this.
        const uint8_t * const inPosRest8 = inPosCol->get_data_uncompr_start();
        const size_t inPosCountLogRest = (inPosSizeComprByte < inPosSizeUsedByte)
                ? convert_size<uint8_t, uint64_t>(inPosSizeUsedByte - (inPosRest8 - inPos))
                : 0;
        const size_t inPosCountLogCompr = inPosCol->get_count_values_compr();

        auto outDataCol = new column<t_out_data_f>(
                get_size_max_byte_any_len<t_out_data_f>(inPosCountLog)
        );
        uint8_t * outData = outDataCol->get_data();
        const uint8_t * const initOutData = outData;
        size_t outDataCountLog;
        size_t outDataSizeComprByte;

        // The state of the nonselective_write_iterator for the compressed
        // output.
        typename project_processing_unit_wit<
                t_ve, t_out_data_f, t_in_data_f
        >::state_t witComprState(outData, inData);
        
        // Processing of the input positions column's compressed part using the
        // specified vector extension, compressed output.
        decompress_and_process_batch<
                t_ve,
                t_in_pos_f,
                project_processing_unit_wit,
                t_out_data_f,
                t_in_data_f
        >::apply(
                inPos, inPosCountLogCompr, witComprState
        );
        
        if(inPosSizeComprByte == inPosSizeUsedByte) {
            // If the input column has no uncompressed rest part, we are done.
            std::tie(
                    outDataSizeComprByte, std::ignore, outData
            ) = witComprState.m_Wit.done();
            outDataCountLog = inPosCountLogCompr;
        }
        else {
            // If the input positions column has an uncompressed rest part.
            
            // Pad the input pointer such that it points to the beginning of
            // the input position column's uncompressed rest part.
            inPos = inPosRest8;
            // The size of the input column's uncompressed rest part.
            //const size_t inPosSizeRestByte =
            //        initInPos + inPosSizeUsedByte - inPos;
            size_t const inPosSizeRestByte = (inPosCol->get_count_values() - inPosCol->get_count_values_compr()) * sizeof(uint64_t);

            // Vectorized processing of the input positions column's
            // uncompressed rest part using the specified vector extension,
            // compressed output.
            const size_t inPosSizeUncomprVecByte = round_down_to_multiple(
                    inPosSizeRestByte, vector_size_byte::value
            );
            decompress_and_process_batch<
                    t_ve,
                    uncompr_f,
                    project_processing_unit_wit,
                    t_out_data_f,
                    t_in_data_f
            >::apply(
                    inPos, convert_size<uint8_t, uint64_t>(inPosSizeUncomprVecByte), witComprState
            );
            
            // Finish the compressed output. This might already initialize the
            // output column's uncompressed rest part.
            uint8_t * outDataAppendUncompr;
            std::tie(
                    outDataSizeComprByte, outDataAppendUncompr, outData
            ) = witComprState.m_Wit.done();
            outDataCountLog = inPosCountLogCompr + round_down_to_multiple(
                    inPosCountLogRest, vector_element_count::value
            );
            
            // The size of the input column's uncompressed rest that can only
            // be processed with scalar instructions.
            const size_t inSizeScalarRemainderByte =
                    inPosSizeRestByte % vector_size_byte::value;
            if(inSizeScalarRemainderByte) {
                // If there is such an uncompressed scalar rest.
                
                // We want to avoid actual scalar processing here, because this
                // would require a scalar implementation of the
                // random_read_access-interface (used inside
                // project_processing_unit_wit) for formats suitable for
                // vectorized processing.
                // Thus, we employ the following workaround: (1) We load the
                // remaining data elements into a vector register and fill the
                // "empty" elements with copies of the last data element in the
                // input position column. (2) We do not use
                // decompress_and_process_batch, but instead call the core
                // operator directly. (3) The output is written to a temporary
                // buffer. From there, we copy only the appropriate number of
                // data elements to the output column's data buffer.
                
                // The number of data elements in the input position column's
                // uncompressed scalar rest.
                const size_t inPosCountLogScalarRemainder =
                        convert_size<uint8_t, base_t>(
                                inSizeScalarRemainderByte
                        );
                
                // Temporary buffer containing the scalar rest and copies of
                // the last data element.
                MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value)
                base_t tmpIn[vector_element_count::value];
                memcpy(tmpIn, inPos, inSizeScalarRemainderByte);
                for(
                        unsigned i = inPosCountLogScalarRemainder;
                        i < vector_element_count::value;
                        i++
                )
                    tmpIn[i] = 0; //tmp[inPosCountLogScalarRemainder - 1];
                
                // Temporary buffer for the output.
                MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value)
                uint8_t tmpOut[vector_size_byte::value];
                
                // The state of the selective write_iterator for the
                // uncompressed output. Note that this uses t_ve, not
                // necessarily scalar and writes to the temporary output
                // buffer.
                typename project_processing_unit_wit<
                        t_ve, uncompr_f, t_in_data_f
                >::state_t witUncomprState(tmpOut, inData);
                
                // Call the core operator directly.
                project_processing_unit_wit<
                        t_ve, uncompr_f, t_in_data_f
                >::apply(
                        load<
                                t_ve, iov::ALIGNED, vector_size_bit::value
                        >(tmpIn),
                        witUncomprState
                );
                
                // Copy only the valid part of the temporary output buffer to
                // the actual output buffer.
                memcpy(outDataAppendUncompr, tmpOut, inSizeScalarRemainderByte);
                
                // Finish the output column's uncompressed rest part.
                outData = outDataAppendUncompr + inSizeScalarRemainderByte;
                outDataCountLog += inPosCountLogScalarRemainder;
            }
        }
        
        // Finish the output column.
        outDataCol->set_meta_data(
                outDataCountLog, outData - initOutData, outDataSizeComprByte
        );

        return outDataCol;
    }
};

// Replication
template<
        class t_vector_extension,
        class t_out_data_f,
        class t_in_pos_f
>
class my_project_repl_wit_t {
    using t_ve = t_vector_extension;
    IMPORT_VECTOR_BOILER_PLATE(t_ve)
public:
    static const column<t_out_data_f> * apply(
            replicated_column * const inDataCol,
            const column<t_in_pos_f> * const inPosCol
    ) {
        using namespace vectorlib;
        void* col;
        size_t format;
        col = inDataCol->get_column<t_ve>(format);
        switch (format)
        {
          case 0: // UNCOMPR
          {
             return my_project_wit_t<t_vector_extension, t_out_data_f, uncompr_f, t_in_pos_f>::apply(
               reinterpret_cast<const column<uncompr_f>*>(col),
               inPosCol
             );
          }
          case 1: // STATICBP 32
          {
             return my_project_wit_t<t_vector_extension, t_out_data_f, DEFAULT_STATIC_VBP_F(t_ve, 32), t_in_pos_f>::apply(
               reinterpret_cast<const column<DEFAULT_STATIC_VBP_F(t_ve, 32)>*>(col),
               inPosCol
             );
          }
          //case 2: // DYNAMICBP
        //{
        //     return my_project_wit_t<t_vector_extension, t_out_data_f, DEFAULT_DYNAMIC_VBP_F(t_ve), t_in_pos_f>::apply(
        //       reinterpret_cast<const column<DEFAULT_DYNAMIC_VBP_F(t_ve)>*>(col),
        //       inPosCol
        //     );
         // }
        }

    }
};

// Replication

}
#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_PROJECT_COMPR_H
