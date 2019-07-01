//
// Created by jpietrzyk on 28.05.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_SEMI_EQUI_COMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_SEMI_EQUI_COMPR_H


#include <core/utils/preprocessor.h>
#include <core/storage/column.h>
#include <core/morphing/format.h>

#include <core/operators/interfaces/join.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <vector/datastructures/hash_based/strategies/linear_probing.h>
#include <vector/datastructures/hash_based/hash_utils.h>
#include <vector/datastructures/hash_based/hash_map.h>
#include <vector/datastructures/hash_based/hash_set.h>
#include <vector/complex/hash.h>

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace morphstore {
   using namespace vector;


   //We assume, that the input column has unique keys!
   template<
      class VectorExtension,
      class DataStructure
   >
   struct semi_equi_join_build_processing_unit_wit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         DataStructure & m_Ds;
         typename DataStructure::template strategy_state< VectorExtension > m_StrategyState;
         state_t(
            DataStructure & p_Ds
         ):
            m_Ds{ p_Ds },
            m_StrategyState{ p_Ds.template get_lookup_insert_strategy_state< VectorExtension >()} {}
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void apply( vector_t const & p_DataVector, state_t & p_State ) {
         p_State.m_Ds.template insert<VectorExtension>(
            p_DataVector,
            p_State.m_StrategyState
         );
      }
   };

   template<
      class VectorExtension,
      class DataStructure,
      class OutFormatCol
   >
   struct semi_equi_join_probe_processing_unit_wit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         DataStructure &   m_Ds;
         vector_t m_Pos;
         vector_t const m_Inc;
         selective_write_iterator<
            VectorExtension,
            OutFormatCol
         >                 m_WitOutData;
         typename DataStructure::template strategy_state< VectorExtension > m_StrategyState;
         state_t(
            DataStructure  &        p_Ds,
            uint8_t        * const  p_OutPtr,
            base_t                  p_Pos
         ):
            m_Ds{ p_Ds },
            m_Pos{ set_sequence< VectorExtension, vector_base_t_granularity::value >( p_Pos, 1 ) },
            m_Inc{ set1< VectorExtension, vector_base_t_granularity::value >( vector_element_count::value ) },
            m_WitOutData{p_OutPtr},
            m_StrategyState{ p_Ds.template get_lookup_insert_strategy_state< VectorExtension >()} {}
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void apply( vector_t const & p_DataVector, state_t & p_State ) {
         vector_mask_t lookupResultMask;
         uint8_t hitResultCount;
         std::tie( lookupResultMask, hitResultCount ) =
            p_State.m_Ds.template lookup<VectorExtension>(
               p_DataVector,
               p_State.m_StrategyState
            );
         p_State.m_WitOutData.write(p_State.m_Pos, lookupResultMask, hitResultCount);
         p_State.m_Pos = add< VectorExtension >::apply( p_State.m_Pos, p_State.m_Inc );
      }
   };

   template<
      class VectorExtension,
      class DataStructure,
      class OutFormatCol,
      class InFormatLCol,
      class InFormatRCol
   >
   struct semi_equi_join_t {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      static
      column< OutFormatCol > const *
      apply(
         column< InFormatLCol > const * const p_InDataLCol,
         column< InFormatRCol > const * const p_InDataRCol,
         size_t const outCountEstimate = 0
      ) {
         uint8_t const *         inBuildDataPtr             = p_InDataLCol->get_data();
         uint8_t const * const   startBuildDataPtr          = inBuildDataPtr;
         uint8_t const *         inProbeDataPtr             = p_InDataRCol->get_data();
         uint8_t const * const   startProbeDataPtr          = inProbeDataPtr;

         size_t  const           inBuildDataCountLog        = p_InDataLCol->get_count_values();
         size_t  const           inProbeDataCountLog        = p_InDataRCol->get_count_values();

         size_t  const           inBuildDataSizeComprByte   = p_InDataLCol->get_size_compr_byte();
         size_t  const           inProbeDataSizeComprByte   = p_InDataRCol->get_size_compr_byte();

         size_t  const           inBuildDataSizeUsedByte    = p_InDataLCol->get_size_used_byte();
         size_t  const           inProbeDataSizeUsedByte    = p_InDataRCol->get_size_used_byte();

         uint8_t const * const   inProbeDataRest8           = create_aligned_ptr(
            inProbeDataPtr + inProbeDataSizeComprByte
         );
         const size_t inProbeCountLogRest = convert_size<uint8_t, uint64_t>(
            inProbeDataSizeUsedByte - (inProbeDataRest8 - inProbeDataPtr)
         );
         const size_t inProbeCountLogCompr = inProbeDataCountLog - inProbeCountLogRest;


         size_t outCountLog;
         DataStructure ds( inBuildDataCountLog );

         auto outPosCol = new column<OutFormatCol>(
            bool(outCountEstimate)
            // use given estimate
            ? get_size_max_byte_any_len<OutFormatCol>(outCountEstimate)
            // use pessimistic estimate
            : get_size_max_byte_any_len<OutFormatCol>(inProbeDataCountLog)
         );


         uint8_t       *         outPtr                     = outPosCol->get_data();
         uint8_t const * const   startOutDataPtr            = outPtr;


         //Build Part Starts Here
         typename semi_equi_join_build_processing_unit_wit<
            VectorExtension,
            DataStructure
         >::state_t witBuildComprState(
            ds
         );
         decompress_and_process_batch<
            VectorExtension,
            InFormatLCol,
            semi_equi_join_build_processing_unit_wit,
            DataStructure
         >::apply(
            inBuildDataPtr, inBuildDataSizeComprByte, witBuildComprState
         );

         if(inBuildDataSizeComprByte != inBuildDataSizeUsedByte) {
            inBuildDataPtr = create_aligned_ptr( inBuildDataPtr );
            size_t const inBuildSizeRestByte = startBuildDataPtr + inBuildDataSizeUsedByte - inBuildDataPtr;

            const size_t inBuildDataSizeUncomprVecByte = round_down_to_multiple(
               inBuildSizeRestByte, vector_size_byte::value
            );

            decompress_and_process_batch<
               VectorExtension,
               uncompr_f,
               semi_equi_join_build_processing_unit_wit,
               DataStructure
            >::apply(
               inBuildDataPtr, inBuildDataSizeUncomprVecByte, witBuildComprState
            );
            const size_t inBuildSizeScalarRemainderByte = inBuildSizeRestByte % vector_size_byte::value;
            if(inBuildSizeScalarRemainderByte) {
               typename semi_equi_join_build_processing_unit_wit<
                  scalar<v64<uint64_t>>,
                  DataStructure
               >::state_t witBuildUncomprState(
                  ds
               );
               decompress_and_process_batch<
                  scalar<v64<uint64_t>>,
                  uncompr_f,
                  semi_equi_join_build_processing_unit_wit,
                  DataStructure
               >::apply(
                  inBuildDataPtr, inBuildSizeScalarRemainderByte, witBuildUncomprState
               );
            }
         }
         //Build Part Ends Here

         //Probe Part Starts Here
         typename semi_equi_join_probe_processing_unit_wit<
            VectorExtension,
            DataStructure,
            OutFormatCol
         >::state_t witProbeComprState(
            ds,
            outPtr,
            0
         );
         decompress_and_process_batch<
            VectorExtension,
            InFormatRCol,
            semi_equi_join_probe_processing_unit_wit,
            DataStructure,
            OutFormatCol
         >::apply(
            inProbeDataPtr, inProbeDataSizeComprByte, witProbeComprState
         );
         size_t outSizeComprByte;

         if( inProbeDataSizeComprByte == inProbeDataSizeUsedByte ) {
            std::tie(
               outSizeComprByte, std::ignore, outPtr
            ) = witProbeComprState.m_WitOutData.done();
            outCountLog = witProbeComprState.m_WitOutData.get_count_values();
         } else {
            inProbeDataPtr = create_aligned_ptr( inProbeDataPtr );
            size_t const inProbeSizeRestByte = startProbeDataPtr + inProbeDataSizeUsedByte - inProbeDataPtr;
            const size_t inProbeDataSizeUncomprVecByte = round_down_to_multiple(
               inProbeSizeRestByte, vector_size_byte::value
            );
            decompress_and_process_batch<
               VectorExtension,
               uncompr_f,
               semi_equi_join_probe_processing_unit_wit,
               DataStructure,
               OutFormatCol
            >::apply(
               inProbeDataPtr, inProbeDataSizeUncomprVecByte, witProbeComprState
            );
            bool outAddedPadding;
            std::tie(
               outSizeComprByte, outAddedPadding, outPtr
            ) = witProbeComprState.m_WitOutData.done();
            outCountLog = witProbeComprState.m_WitOutData.get_count_values();

            const size_t inProbeSizeScalarRemainderByte = inProbeSizeRestByte % vector_size_byte::value;
            if(inProbeSizeScalarRemainderByte) {
               if(!outAddedPadding)
                  outPtr = create_aligned_ptr(outPtr);
               typename semi_equi_join_probe_processing_unit_wit<
                  scalar<v64<uint64_t>>,
                  DataStructure,
                  uncompr_f
               >::state_t witProbeUncomprState(
                  ds,
                  outPtr,
                  inProbeCountLogCompr + inProbeDataSizeUncomprVecByte / sizeof(base_t)
               );
               decompress_and_process_batch<
                  scalar<v64<uint64_t>>,
                  uncompr_f,
                  semi_equi_join_probe_processing_unit_wit,
                  DataStructure,
                  uncompr_f
               >::apply(
                  inProbeDataPtr, inProbeSizeScalarRemainderByte, witProbeUncomprState
               );

               std::tie(
                  std::ignore, std::ignore, outPtr
               ) = witProbeUncomprState.m_WitOutData.done();
               outCountLog += witProbeComprState.m_WitOutData.get_count_values();
            }
         }
         //Probe Part Ends Here
         outPosCol->set_meta_data(
            outCountLog,  outPtr - startOutDataPtr, outSizeComprByte
         );
         return outPosCol;
      }
   };

}
#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_SEMI_EQUI_COMPR_H
