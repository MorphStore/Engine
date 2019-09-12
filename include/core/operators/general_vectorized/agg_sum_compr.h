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
 * @file agg_sum_compr.h
 * @brief Whole-column aggregation-operator based on the vector-lib, weaving
 * the operator's core into the decompression routine of the input data's
 * format.
 * @todo Currently, it is not truly general-vectorized, because the current
 * implementation of decompress_and_process_batch is hand-written scalar code.
 * @todo Support columns of arbitrary length w.r.t. vectorization and
 * compression.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_AGG_SUM_COMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_AGG_SUM_COMPR_H

#include <core/morphing/format.h>
#include <core/operators/interfaces/agg_sum.h>

#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <cstdint>

namespace morphstore {
   using namespace vectorlib;

   template<
      class VectorExtension
   >
   struct agg_sum_processing_unit_wit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         vector_t m_Aggregate;
         state_t(
            base_t      const p_Aggregate
         ):
            m_Aggregate{ set1<VectorExtension, vector_base_t_granularity::value>( p_Aggregate ) } { }
      };

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void apply(
         vector_t const & p_DataVector,
         state_t        & p_State
      ) {
         p_State.m_Aggregate = add< VectorExtension, vector_base_t_granularity::value >::apply(
            p_State.m_Aggregate, p_DataVector
         );
      }
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static base_t finalize(
         state_t        & p_State
      ) {
         return hadd< VectorExtension, vector_base_t_granularity::value >::apply( p_State.m_Aggregate );
      }
   };

   template<
      class VectorExtension,
      class InFormatCol
   >
   struct agg_sum_t {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      static
      column< uncompr_f > const *
      apply(
         column< InFormatCol > const * const p_InDataCol
      ) {
         uint8_t const *         inDataPtr            = p_InDataCol->get_data();
         uint8_t const * const   startDataPtr         = inDataPtr;
//         size_t  const           inDataCountLog       = p_InDataCol->get_count_values();
         size_t  const           inDataSizeComprByte  = p_InDataCol->get_size_compr_byte();
         size_t  const           inDataSizeUsedByte   = p_InDataCol->get_size_used_byte();
//         uint8_t const * const   inDataRest8          = create_aligned_ptr(
//            inDataPtr + inDataSizeComprByte
//         );
//         size_t  const           inCountLogRest       = convert_size<uint8_t, uint64_t>(
//            inDataSizeUsedByte - ( inDataRest8 - inDataPtr )
//         );
//         size_t  const           inCountLogCompr      = inDataCountLog - inCountLogRest;

         auto outCol = new column< uncompr_f >( sizeof( base_t ) );
         base_t        *         outPtr               = outCol->get_data();

         typename agg_sum_processing_unit_wit<
            VectorExtension
         >::state_t witState(
            0
         );
         decompress_and_process_batch<
            VectorExtension,
            InFormatCol,
            agg_sum_processing_unit_wit
         >::apply(
            inDataPtr, p_InDataCol->get_count_values_compr(), witState
         );
         if(inDataSizeComprByte == inDataSizeUsedByte ) {
            *outPtr = agg_sum_processing_unit_wit<VectorExtension>::finalize( witState );
         } else {
            inDataPtr = create_aligned_ptr( inDataPtr );
            size_t const inSizeRestByte = startDataPtr + inDataSizeUsedByte - inDataPtr;
            size_t const inDataSizeUncomprVecByte = round_down_to_multiple(
               inSizeRestByte, vector_size_byte::value
            );
            decompress_and_process_batch<
               VectorExtension,
               uncompr_f,
               agg_sum_processing_unit_wit
            >::apply(
               inDataPtr,
               convert_size<uint8_t, uint64_t>(inDataSizeUncomprVecByte),
               witState
            );
            size_t const inSizeScalarRemainderByte = inSizeRestByte % vector_size_byte::value;
            base_t result = agg_sum_processing_unit_wit<VectorExtension>::finalize( witState );
            if( inSizeScalarRemainderByte ) {
               typename agg_sum_processing_unit_wit<
                  scalar<v64<uint64_t>>
               >::state_t witUncomprState(
                  result
               );
               decompress_and_process_batch<
                  scalar<v64<uint64_t>>,
                  uncompr_f,
                  agg_sum_processing_unit_wit
               >::apply(
                  inDataPtr,
                  convert_size<uint8_t, uint64_t>(inSizeScalarRemainderByte),
                  witUncomprState
               );
               result = agg_sum_processing_unit_wit<scalar<v64<uint64_t>>>::finalize( witUncomprState );
            }
            *outPtr = result;
         }
         outCol->set_meta_data(
            1,  sizeof( uint64_t ), sizeof( uint64_t )
         );
         return  outCol;
      }
   };
   
}

#endif /* MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_AGG_SUM_COMPR_H */

