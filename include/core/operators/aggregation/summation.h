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
 * @file summation.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_AGGREGATION_LOOKUP_H
#define MORPHSTORE_CORE_OPERATORS_AGGREGATION_LOOKUP_H

#include <core/storage/column.h>

#include "immintrin.h"

namespace morphstore {

uint64_t aggregate_sum(
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
