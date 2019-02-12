/**
 * @file summation.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_AGGREGATION_LOOKUP_H
#define MORPHSTORE_CORE_OPERATORS_AGGREGATION_LOOKUP_H

#include "../../storage/column.h"

#include "immintrin.h"

namespace morphstore {

uint64_t aggregate_sum(
   column< uncompr_f > const * const p_DataColumn
) {

   uint64_t result = 0;
   size_t const elementsPerVectorCount = sizeof( __m128i ) / sizeof( uint64_t );

   size_t const vectorCount = p_DataColumn->count_values() / elementsPerVectorCount;
   size_t const remainderCount = p_DataColumn->count_values() % elementsPerVectorCount;
   __m128i const * dataVecPtr = p_DataColumn->data( );
   __m128i resultVec = _mm_setzero_si128( );
   for( size_t i = 0; i < vectorCount; ++i ) {
      resultVec = _mm_add_epi64( resultVec, _mm_load_si128( dataVecPtr++ ) );
   }

   uint64_t resultArray[ elementsPerVectorCount ];
   _mm_store_si128( reinterpret_cast< __m128i * >( &resultArray ), resultVec );

   for( size_t i = 0; i < elementsPerVectorCount; ++i ) {
      result += resultArray[ i ];
   }

   if( remainderCount != 0) {
      uint64_t const * remainderPtr = reinterpret_cast< uint64_t const * >( dataVecPtr );
      for( size_t i = 0; i < remainderCount; ++i ) {
         result += *remainderPtr++;
      }
   }
   return result;

}

}

#endif //MORPHSTORE_CORE_OPERATORS_AGGREGATION_LOOKUP_H
