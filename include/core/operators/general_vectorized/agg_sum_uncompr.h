//
// Created by jpietrzyk on 26.04.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_AGG_SUM_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_AGG_SUM_UNCOMPR_H

#include <vector/general_vector.h>
#include <vector/primitives/calc.h>
#include <vector/primitives/io.h>
#include <vector/primitives/create.h>

namespace morphstore {

   template<class VectorExtension>
   const column<uncompr_f> *
      agg_sum(
      column< uncompr_f > const * const p_DataColumn
   ) {
      using namespace vector;

      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

      vector_base_t result = 0;

      size_t const vectorCount = p_DataColumn->get_count_values() / vector_element_count::value;
      size_t const remainderCount = p_DataColumn->get_count_values() % vector_element_count::value;
      vector_base_t const * dataPtr = p_DataColumn->get_data( );
      vector_t resultVec = set1<VectorExtension,vector_base_type_size_bit::value>(0);// = setzero<VectorExtension>( );
      alignas(vector_alignment::value) vector_base_t tmp[ vector_element_count::value ];
      for( size_t i = 0; i < vectorCount; ++i ) {
         resultVec = add<VectorExtension, vector_base_t_granularity::value>(
            resultVec, load<VectorExtension,iov::ALIGNED, vector_size_bit::value>( dataPtr )
         );
         store<VectorExtension,iov::ALIGNED, vector_size_bit::value>( tmp, resultVec );
         dataPtr += vector_element_count::value;
      }

      alignas(vector_alignment::value) vector_base_t resultArray[ vector_element_count::value ];
      store<VectorExtension,iov::ALIGNED, vector_size_bit::value>( resultArray, resultVec );

      for( size_t i = 0; i < vector_element_count::value; ++i ) {
         result += resultArray[ i ];
      }

      if( remainderCount != 0) {
         vector_base_t const * remainderPtr = dataPtr;
         for( size_t i = 0; i < remainderCount; ++i ) {
            result += *remainderPtr++;
         }
      }

      auto outDataCol = new column<uncompr_f>(sizeof(vector_base_t));
      vector_base_t * const outData = outDataCol->get_data();
      *outData=result;
      outDataCol->set_meta_data(1, sizeof(vector_base_t));
      return outDataCol;
   }

}



#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_AGG_SUM_UNCOMPR_H

