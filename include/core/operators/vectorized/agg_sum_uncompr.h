/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   agg_sum_uncompr.h
 * Author: Annett
 *
 * Created on 19. März 2019, 11:04
 */

#ifndef AGG_SUM_UNCOMPR_H
#define AGG_SUM_UNCOMPR_H

#include <core/operators/interfaces/agg_sum.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/sse/extension_sse.h>

#include <immintrin.h>

namespace morphstore {
    
template<>
const column<uncompr_f> *
agg_sum<vector::sse<vector::v128<uint64_t>>>(
   column< uncompr_f > const * const p_DataColumn
) {

   uint64_t result = 0;
   size_t const elementsPerVectorCount = sizeof( __m128i ) / sizeof( uint64_t );

   size_t const vectorCount = p_DataColumn->get_count_values() / elementsPerVectorCount;
   size_t const remainderCount = p_DataColumn->get_count_values() % elementsPerVectorCount;
   __m128i const * dataVecPtr = p_DataColumn->get_data( );
   __m128i resultVec = _mm_setzero_si128( );
   for( size_t i = 0; i < vectorCount; ++i ) {
      resultVec = _mm_add_epi64( resultVec, _mm_load_si128( dataVecPtr++ ) );
   }

   alignas(64) uint64_t resultArray[ elementsPerVectorCount ];
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
   
   auto outDataCol = new column<uncompr_f>(sizeof(uint64_t));
   uint64_t * const outData = outDataCol->get_data();
   *outData=result;
   outDataCol->set_meta_data(1, sizeof(uint64_t));
   return outDataCol;
   //return result;

}


template<>
const column<uncompr_f> *
agg_sum<vector::avx2<vector::v256<uint64_t>>>(
   column< uncompr_f > const * const p_DataColumn
) {

   uint64_t result = 0;
   size_t const elementsPerVectorCount = sizeof( __m256i ) / sizeof( uint64_t );

   size_t const vectorCount = p_DataColumn->get_count_values() / elementsPerVectorCount;
   size_t const remainderCount = p_DataColumn->get_count_values() % elementsPerVectorCount;
   __m256i const * dataVecPtr = p_DataColumn->get_data( );
   __m256i resultVec = _mm256_setzero_si256( );
   for( size_t i = 0; i < vectorCount; ++i ) {
      resultVec = _mm256_add_epi64( resultVec, _mm256_load_si256( dataVecPtr++ ) );
   }

  debug( "elpvc: " , elementsPerVectorCount);
   alignas(64) uint64_t resultArray[ elementsPerVectorCount ];
   for (int i=0;i<4;i++) {
       resultArray[i]=0;
       debug( "resultarray " ,i, ": ", resultArray[i]);
       
   }
   debug("resVec: ", 0, ": ", _mm256_extract_epi64(resultVec, 0) );
   debug("resVec: ", 1, ": ", _mm256_extract_epi64(resultVec, 1) );
   debug("resVec: ", 2, ": ", _mm256_extract_epi64(resultVec, 2) );
   debug("resVec: ", 3, ": ", _mm256_extract_epi64(resultVec, 3) );
   

   _mm256_store_si256(reinterpret_cast< __m256i * >( &resultArray ) , resultVec );

   for( size_t i = 0; i < elementsPerVectorCount; ++i ) {
      result += resultArray[ i ];
   }

   if( remainderCount != 0) {
      uint64_t const * remainderPtr = reinterpret_cast< uint64_t const * >( dataVecPtr );
      for( size_t i = 0; i < remainderCount; ++i ) {
         result += *remainderPtr++;
      }
   }
   //return result;
   auto outDataCol = new column<uncompr_f>(sizeof(uint64_t));
   uint64_t * const outData = outDataCol->get_data();
   *outData=result;
   outDataCol->set_meta_data(1, sizeof(uint64_t));
   return outDataCol;

}

/*
uint64_t aggregate_sum_512(
   column< uncompr_f > const * const p_DataColumn
) {

   uint64_t result = 0;
   size_t const elementsPerVectorCount = sizeof( __m512i ) / sizeof( uint64_t );

   size_t const vectorCount = p_DataColumn->get_count_values() / elementsPerVectorCount;
   size_t const remainderCount = p_DataColumn->get_count_values() % elementsPerVectorCount;
   __m512i const * dataVecPtr = p_DataColumn->get_data( );
   __m512i resultVec = _mm512_setzero_si512( );
   for( size_t i = 0; i < vectorCount; ++i ) {
      resultVec = _mm512_add_epi64( resultVec, _mm512_load_si512( dataVecPtr++ ) );
   }

   uint64_t resultArray[ elementsPerVectorCount ];
   _mm512_store_si512( reinterpret_cast< __m512i * >( &resultArray ), resultVec );

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

}*/

}

#endif /* AGG_SUM_UNCOMPR_H */

