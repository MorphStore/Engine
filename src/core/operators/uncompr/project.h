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


#ifndef MORPHSTORE_CORE_OPERATORS_UNCOMPR_PROJECT_H
#define MORPHSTORE_CORE_OPERATORS_UNCOMPR_PROJECT_H

#include <core/utils/preprocessor.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

namespace morphstore {
    using namespace vectorlib;
    
    template< class t_vector_extension >
    struct project_processing_unit {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static vector_t
        apply(
          base_t const * p_DataPtr,
          vector_t p_PosVector
        ) {
            //@todo: Is it better to avoid gather here?
            return
              vectorlib::gather<t_vector_extension, vector_size_bit::value, sizeof(base_t)>(p_DataPtr, p_PosVector);
        }
    };
    
    
    template< class t_vector_extension >
    struct project_batch {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static void
        apply(
          base_t const *& p_DataPtr,
          base_t const *& p_PosPtr,
          base_t *& p_OutPtr,
          size_t const p_Count
        ) {
            for (size_t i = 0; i < p_Count; ++ i) {
                vector_t posVector
                  = vectorlib::load<t_vector_extension, vectorlib::iov::ALIGNED, vector_size_bit::value>(p_PosPtr);
                vector_t lookupVector
                  = project_processing_unit<t_vector_extension>::apply(p_DataPtr, posVector);
                vectorlib
                  ::store<t_vector_extension, vectorlib::iov::ALIGNED, vector_size_bit::value>
                    (p_OutPtr, lookupVector);
                p_PosPtr += vector_element_count::value;
                p_OutPtr += vector_element_count::value;
            }
        }
    };
    
    
    template< class t_vector_extension, class t_out_data_f, class t_in_data_f, class t_in_pos_f >
    struct project_t {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static column <t_out_data_f> const *
        apply(
          column <t_in_data_f> const * const p_DataColumn,
          column <t_in_pos_f> const * const p_PosColumn
        ) {
            size_t const inPosCount = p_PosColumn->get_count_values();
            size_t const inUsedBytes = p_PosColumn->get_size_used_byte();
            size_t const vectorCount = inPosCount / vector_element_count::value;
            size_t const remainderCount = inPosCount % vector_element_count::value;
            base_t const * inDataPtr = p_DataColumn->get_data();
            base_t const * inPosPtr = p_PosColumn->get_data();
            auto outDataCol = new column<uncompr_f>(inUsedBytes);
            base_t * outDataPtr = outDataCol->get_data();
            
            project_batch<t_vector_extension>::apply(inDataPtr, inPosPtr, outDataPtr, vectorCount);
            project_batch<scalar<v64<uint64_t>>>::apply(inDataPtr, inPosPtr, outDataPtr, remainderCount);
            
            outDataCol->set_meta_data(inPosCount, inUsedBytes);
            
            return outDataCol;
        }
    };
    
} /// namespace morphstore

#endif // MORPHSTORE_CORE_OPERATORS_UNCOMPR_PROJECT_H

