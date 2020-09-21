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
 * @file agg_sum.h
 * @brief 
 */

#ifndef MORPHSTORE_CORE_OPERATORS_UNCOMPR_AGG_SUM_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_UNCOMPR_AGG_SUM_UNCOMPR_H

#include <core/operators/interfaces/agg_sum.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

namespace morphstore {

   using namespace vectorlib;
   
   /*template<class VectorExtension>
   const column<uncompr_f> *
      agg_sum(
      column< uncompr_f > const * const p_DataColumn
   ) {
      using namespace vectorlib;

      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

      size_t const vectorCount = p_DataColumn->get_count_values() / vector_element_count::value;
      size_t const remainderCount = p_DataColumn->get_count_values() % vector_element_count::value;
      base_t const * dataPtr = p_DataColumn->get_data( );
      vector_t resultVec = set1<VectorExtension,base_type_size_bit::value>(0);// = setzero<VectorExtension>( );
      for( size_t i = 0; i < vectorCount; ++i ) {
         resultVec = add<VectorExtension, base_t_granularity::value>(
            resultVec, load<VectorExtension,iov::ALIGNED, vector_size_bit::value>( dataPtr )
         );
         dataPtr += vector_element_count::value;
      }

      base_t result = hadd<VectorExtension,vector_base_type_size_bit::value>( resultVec );

      if( remainderCount != 0) {
         base_t const * remainderPtr = dataPtr;
         for( size_t i = 0; i < remainderCount; ++i ) {
            result += *remainderPtr++;
         }
      }

      auto outDataCol = new column<uncompr_f>(sizeof(base_t));
      base_t * const outData = outDataCol->get_data();
      *outData=result;
      outDataCol->set_meta_data(1, sizeof(base_t));
      return outDataCol;
   }*/

   template<class VectorExtension>
   struct agg_sum_processing_unit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         vector_t resultVec;
         state_t(void): resultVec( vectorlib::set1<VectorExtension, vector_base_t_granularity::value>( 0 ) ) { }
         //state_t(vector_t const & p_Data): resultVec( p_Data ) { }
         state_t(base_t p_Data): resultVec(vectorlib::set1<scalar<v64<uint64_t>>,64>(p_Data)){}
         //TODO replace by set
      };
      
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
         vector_t const & p_DataVector,
         state_t & p_State
      ) {
         p_State.resultVec = vectorlib::add<VectorExtension, vector_base_t_granularity::value>::apply(
            p_State.resultVec, p_DataVector
         );
      }
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static base_t finalize(
         typename agg_sum_processing_unit<VectorExtension>::state_t & p_State
      ) {
          
         return vectorlib::hadd<VectorExtension,vector_base_t_granularity::value>::apply( p_State.resultVec );
      }
   };

   template<class VectorExtension>
   struct agg_sum_batch {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static
      base_t
      apply(
         base_t const *& p_DataPtr,
         size_t const p_Count,
         typename agg_sum_processing_unit<VectorExtension>::state_t &p_State
      ) {
         for(size_t i = 0; i < p_Count; ++i) {
            vector_t dataVector = vectorlib::load<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(p_DataPtr);
            agg_sum_processing_unit<VectorExtension>::apply(
               dataVector,
               p_State
            );
            p_DataPtr += vector_element_count::value;
         }
         return agg_sum_processing_unit<VectorExtension>::finalize( p_State );
      }
   };

   template<class t_vector_extension>
   struct agg_sum_all_t<t_vector_extension, uncompr_f, uncompr_f> {
      IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
      
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      const column<uncompr_f> *
      apply(column< uncompr_f > const * const p_DataColumn) {
         typename agg_sum_processing_unit<t_vector_extension>::state_t vectorState;
         size_t const vectorCount = p_DataColumn->get_count_values() / vector_element_count::value;
         size_t const remainderCount = p_DataColumn->get_count_values() % vector_element_count::value;
         base_t const * dataPtr = p_DataColumn->get_data( );

         base_t t=agg_sum_batch<t_vector_extension>::apply( dataPtr, vectorCount, vectorState );
         typename agg_sum_processing_unit<scalar<v64<uint64_t>>>::state_t scalarState(
         t
         );

        base_t result =
            agg_sum_batch<scalar < v64 < uint64_t > >>::apply( dataPtr, remainderCount, scalarState );

         auto outDataCol = new column<uncompr_f>(sizeof(base_t));
         base_t * const outData = outDataCol->get_data();
         *outData = result;
         outDataCol->set_meta_data(1, sizeof(base_t));
         return outDataCol;
      }
   };
   
 
}



#endif //MORPHSTORE_CORE_OPERATORS_UNCOMPR_AGG_SUM_UNCOMPR_H

