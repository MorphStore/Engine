//
// Created by jpietrzyk on 28.05.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_NATURAL_EQUI_COMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_NATURAL_EQUI_COMPR_H


#include <core/utils/preprocessor.h>
#include <core/storage/column.h>
#include <core/morphing/format.h>
#include <core/morphing/write_iterator.h>

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

//
#include <core/morphing/default_formats.h>
#include <core/storage/replicated_column.h>

namespace morphstore {
   using namespace vectorlib;


   template<
      class VectorExtension,
      class DataStructure
   >
   struct natural_equi_join_build_processing_unit_wit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         DataStructure & m_Ds;
         vector_t m_BuildPos;
         vector_t const m_Inc;
         typename DataStructure::template strategy_state< VectorExtension > m_StrategyState;
         state_t(
            DataStructure &         p_Ds,
            base_t          const   p_BuildPos
         ):
            m_Ds{ p_Ds },
            m_BuildPos{ set_sequence< VectorExtension, vector_base_t_granularity::value >( p_BuildPos, 1 ) },
            m_Inc{ set1< VectorExtension, vector_base_t_granularity::value >( vector_element_count::value ) },
            m_StrategyState{ p_Ds.template get_lookup_insert_strategy_state< VectorExtension >()} {}
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void apply( vector_t const & p_DataVector, state_t & p_State ) {
         p_State.m_Ds.template insert<VectorExtension>(
            p_DataVector,
            p_State.m_BuildPos,
            p_State.m_StrategyState
         );
         p_State.m_BuildPos = add< VectorExtension >::apply( p_State.m_BuildPos, p_State.m_Inc );
      }
   };

   template<
      class VectorExtension,
      class DataStructure,
      class OutFormatLCol,
      class OutFormatRCol
   >
   struct natural_equi_join_probe_processing_unit_wit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         DataStructure & m_Ds;
         vector_t          m_ProbePos;
         vector_t const    m_Inc;
         typename DataStructure::template strategy_state< VectorExtension >
                           m_StrategyState;
         selective_write_iterator<
            VectorExtension,
            OutFormatLCol
         >                 m_WitOutLData;
         selective_write_iterator<
            VectorExtension,
            OutFormatRCol
         >                 m_WitOutRData;
         state_t(
            DataStructure  &        p_Ds,
            uint8_t        * const  p_OutLData,
            uint8_t        * const  p_OutRData,
            base_t           const  p_ProbePos
         ):
            m_Ds{ p_Ds },
            m_ProbePos{ set_sequence< VectorExtension, vector_base_t_granularity::value >( p_ProbePos, 1 ) },
            m_Inc{ set1< VectorExtension, vector_base_t_granularity::value >( vector_element_count::value ) },
            m_StrategyState{ p_Ds.template get_lookup_insert_strategy_state< VectorExtension >()},
            m_WitOutLData{ p_OutLData },
            m_WitOutRData{ p_OutRData } {}
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void apply( vector_t const & p_DataVector, state_t & p_State ) {
         vector_t lookupResultValuesVector;
         vector_mask_t lookupResultMask;
         uint8_t hitResultCount;
         std::tie( lookupResultValuesVector, lookupResultMask, hitResultCount ) =
            p_State.m_Ds.template lookup<VectorExtension>(
               p_DataVector,
               p_State.m_StrategyState
            );
         p_State.m_WitOutLData.write(lookupResultValuesVector, lookupResultMask, hitResultCount);
         p_State.m_WitOutRData.write(p_State.m_ProbePos, lookupResultMask, hitResultCount);
         p_State.m_ProbePos = add< VectorExtension >::apply( p_State.m_ProbePos, p_State.m_Inc );
      }
   };


   template<
      class VectorExtension,
      class DataStructure,
      class OutFormatLCol,
      class OutFormatRCol,
      class InFormatLCol,
      class InFormatRCol
   >
   struct natural_equi_join_t {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      static
      std::tuple<
         column< OutFormatLCol > const *,
      column< OutFormatRCol > const *
      > const
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

         uint8_t const * const   inProbeDataRest8           = p_InDataRCol->get_data_uncompr_start();
         const size_t inProbeCountLogRest                   = convert_size<uint8_t, uint64_t>(
            inProbeDataSizeUsedByte - (inProbeDataRest8 - inProbeDataPtr) //
         );
         const size_t inProbeCountLogCompr = inProbeDataCountLog - inProbeCountLogRest;


         uint8_t const * const   inBuildDataRest8           = p_InDataLCol->get_data_uncompr_start();
         const size_t inBuildCountLogRest                   = convert_size<uint8_t, uint64_t>(
            inBuildDataSizeUsedByte - (inBuildDataRest8 - inBuildDataPtr)// get uncomp
         );
         const size_t inBuildCountLogCompr = inBuildDataCountLog - inBuildCountLogRest;


         size_t outCountLog;
         DataStructure ds( inBuildDataCountLog );

         auto outPosLCol = new column<OutFormatLCol>(
            bool(outCountEstimate)
            // use given estimate
            ? get_size_max_byte_any_len<OutFormatLCol>(outCountEstimate)
            // use pessimistic estimate
            : get_size_max_byte_any_len<OutFormatLCol>(inProbeDataCountLog)
         );
         auto outPosRCol = new column<OutFormatRCol>(
            bool(outCountEstimate)
            // use given estimate
            ? get_size_max_byte_any_len<OutFormatRCol>(outCountEstimate)
            // use pessimistic estimate
            : get_size_max_byte_any_len<OutFormatRCol>(inProbeDataCountLog)
         );


         uint8_t       *         outLPtr                     = outPosLCol->get_data();
         uint8_t const * const   startOutLDataPtr            = outLPtr;
         uint8_t       *         outRPtr                     = outPosRCol->get_data();
         uint8_t const * const   startOutRDataPtr            = outRPtr;


         //Build Part Starts Here
         typename natural_equi_join_build_processing_unit_wit<
            VectorExtension,
            DataStructure
         >::state_t witBuildComprState(
            ds,
            0
         );
         decompress_and_process_batch<
            VectorExtension,
            InFormatLCol,
            natural_equi_join_build_processing_unit_wit,
            DataStructure
         >::apply(
            inBuildDataPtr,
            p_InDataLCol->get_count_values_compr(),
            witBuildComprState
         );

         if(inBuildDataSizeComprByte != inBuildDataSizeUsedByte) {
            inBuildDataPtr = inBuildDataRest8;
            //size_t const inBuildSizeRestByte = startBuildDataPtr + inBuildDataSizeUsedByte - inBuildDataPtr;
            size_t const inBuildSizeRestByte = (p_InDataLCol->get_count_values() - p_InDataLCol->get_count_values_compr()) * sizeof(uint64_t);

            const size_t inBuildDataSizeUncomprVecByte = round_down_to_multiple(
               inBuildSizeRestByte, vector_size_byte::value
            );

            decompress_and_process_batch<
               VectorExtension,
               uncompr_f,
               natural_equi_join_build_processing_unit_wit,
               DataStructure
            >::apply(
               inBuildDataPtr,
               convert_size<uint8_t, uint64_t>(inBuildDataSizeUncomprVecByte),
               witBuildComprState
            );
            const size_t inBuildSizeScalarRemainderByte = inBuildSizeRestByte % vector_size_byte::value;
            if(inBuildSizeScalarRemainderByte) {
               typename natural_equi_join_build_processing_unit_wit<
                  scalar<v64<uint64_t>>,
                  DataStructure
               >::state_t witBuildUncomprState(
                  ds,
                  inBuildCountLogCompr + inBuildDataSizeUncomprVecByte / sizeof(base_t)
               );
               decompress_and_process_batch<
                  scalar<v64<uint64_t>>,
                  uncompr_f,
                  natural_equi_join_build_processing_unit_wit,
                  DataStructure
               >::apply(
                  inBuildDataPtr,
                  convert_size<uint8_t, uint64_t>(
                     inBuildSizeScalarRemainderByte
                  ),
                  witBuildUncomprState
               );
            }
         }
         //Build Part Ends Here

         //Probe Part Starts Here
         typename natural_equi_join_probe_processing_unit_wit<
            VectorExtension,
            DataStructure,
            OutFormatLCol,
            OutFormatRCol
         >::state_t witProbeComprState(
            ds,
            outLPtr,
            outRPtr,
            0
         );
         decompress_and_process_batch<
            VectorExtension,
            InFormatRCol,
            natural_equi_join_probe_processing_unit_wit,
            DataStructure,
            OutFormatLCol,
            OutFormatRCol
         >::apply(
            inProbeDataPtr,
            p_InDataRCol->get_count_values_compr(),
            witProbeComprState
         );
         size_t outSizeLComprByte;
         size_t outSizeRComprByte;

         if( inProbeDataSizeComprByte == inProbeDataSizeUsedByte ) {
            std::tie(
               outSizeLComprByte, std::ignore, outLPtr
            ) = witProbeComprState.m_WitOutLData.done();
            std::tie(
               outSizeRComprByte, std::ignore, outRPtr
            ) = witProbeComprState.m_WitOutRData.done();
            outCountLog = witProbeComprState.m_WitOutLData.get_count_values();
         } else {
            inProbeDataPtr = inProbeDataRest8;
            //size_t const inProbeSizeRestByte = startProbeDataPtr + inProbeDataSizeUsedByte - inProbeDataPtr;
            size_t const inProbeSizeRestByte = (p_InDataRCol->get_count_values() - p_InDataRCol->get_count_values_compr()) * sizeof(uint64_t);
            const size_t inProbeDataSizeUncomprVecByte = round_down_to_multiple(
               inProbeSizeRestByte, vector_size_byte::value
            );
            decompress_and_process_batch<
               VectorExtension,
               uncompr_f,
               natural_equi_join_probe_processing_unit_wit,
               DataStructure,
               OutFormatLCol,
               OutFormatRCol
            >::apply(
               inProbeDataPtr,
               convert_size<uint8_t, uint64_t>(inProbeDataSizeUncomprVecByte),
               witProbeComprState
            );
            uint8_t * outLAppendUncompr;
            uint8_t * outRAppendUncompr;
            std::tie(
               outSizeLComprByte, outLAppendUncompr, outLPtr
            ) = witProbeComprState.m_WitOutLData.done();
            std::tie(
               outSizeRComprByte, outRAppendUncompr, outRPtr
            ) = witProbeComprState.m_WitOutRData.done();
            outCountLog = witProbeComprState.m_WitOutLData.get_count_values();

            const size_t inProbeSizeScalarRemainderByte = inProbeSizeRestByte % vector_size_byte::value;
            if(inProbeSizeScalarRemainderByte) {
               typename natural_equi_join_probe_processing_unit_wit<
                  scalar<v64<uint64_t>>,
                  DataStructure,
                  uncompr_f,
                  uncompr_f
               >::state_t witProbeUncomprState(
                  ds,
                  outLAppendUncompr,
                  outRAppendUncompr,
                  inProbeCountLogCompr + inProbeDataSizeUncomprVecByte / sizeof(base_t)
               );
               decompress_and_process_batch<
                  scalar<v64<uint64_t>>,
                  uncompr_f,
                  natural_equi_join_probe_processing_unit_wit,
                  DataStructure,
                  uncompr_f,
                  uncompr_f
               >::apply(
                  inProbeDataPtr,
                  convert_size<uint8_t, uint64_t>(
                     inProbeSizeScalarRemainderByte
                  ),
                  witProbeUncomprState
               );

               std::tie(
                  std::ignore, std::ignore, outLPtr
               ) = witProbeUncomprState.m_WitOutLData.done();
               std::tie(
                  std::ignore, std::ignore, outRPtr
               ) = witProbeUncomprState.m_WitOutRData.done();
               outCountLog += witProbeUncomprState.m_WitOutLData.get_count_values();
            }
         }
         //Probe Part Ends Here
         outPosLCol->set_meta_data(
            outCountLog,  outLPtr - startOutLDataPtr, outSizeLComprByte
         );
         outPosRCol->set_meta_data(
            outCountLog,  outRPtr - startOutRDataPtr, outSizeRComprByte
         );
         return std::make_tuple( outPosLCol, outPosRCol );
      }
   };

// Replication

template<
      class VectorExtension,
      class DataStructure,
      class OutFormatLCol,
      class OutFormatRCol,
      class InFormatCol
>
   struct natural_equi_join_repl_t {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

      // Left column is a replicated one
      static
      std::tuple<
      column< OutFormatLCol > const *,
      column< OutFormatRCol > const *
      > const
      apply(
         replicated_column * const p_InDataLCol,
         column< InFormatCol > const * const p_InDataRCol,
         size_t const outCountEstimate = 0
      ) {
        void* col;
        size_t format;
        col = p_InDataLCol->get_column<VectorExtension>(format);
        switch (format)
        {
          case 0: // UNCOMPR
          {
             //return natural_equi_join_t<VectorExtension, DataStructure, OutFormatLCol, OutFormatRCol, uncompr_f, InFormatCol>::apply(
             return join<VectorExtension, OutFormatLCol, OutFormatRCol, uncompr_f, InFormatCol>(
               reinterpret_cast<const column<uncompr_f>*>(col),
               p_InDataRCol,
               outCountEstimate
             );
          }
          case 1: // STATICBP 32
          {
             //return natural_equi_join_t<VectorExtension, DataStructure, OutFormatLCol, OutFormatRCol, DEFAULT_STATIC_VBP_F(VectorExtension, 32), InFormatCol>::apply(
             return join<VectorExtension, OutFormatLCol, OutFormatRCol, DEFAULT_STATIC_VBP_F(VectorExtension, 32), InFormatCol>(
               reinterpret_cast<const column<DEFAULT_STATIC_VBP_F(VectorExtension, 32)>*>(col),
               p_InDataRCol,
               outCountEstimate
             );
          }
          case 2: // DYNAMICBP
          {
             //return natural_equi_join_t<VectorExtension, DataStructure, OutFormatLCol, OutFormatRCol, DEFAULT_DYNAMIC_VBP_F(VectorExtension), InFormatCol>::apply(
             return join<VectorExtension, OutFormatLCol, OutFormatRCol, DEFAULT_DYNAMIC_VBP_F(VectorExtension), InFormatCol>(
               reinterpret_cast<const column<DEFAULT_DYNAMIC_VBP_F(VectorExtension)>*>(col),
               p_InDataRCol,
               outCountEstimate
             );
          }
        }
      }

      // Right column is a replicated one
      static
      std::tuple<
      column< OutFormatLCol > const *,
      column< OutFormatRCol > const *
      > const
      apply(
         column< InFormatCol > const * const p_InDataLCol,
         replicated_column * const p_InDataRCol,
         size_t const outCountEstimate = 0
      ) {
        void* col;
        size_t format;
        col = p_InDataRCol->get_column<VectorExtension>(format);
        switch (format)
        {
          case 0: // UNCOMPR
          {
             //return natural_equi_join_t<VectorExtension, DataStructure, OutFormatLCol, OutFormatRCol, InFormatCol, uncompr_f>::apply(
             return join<VectorExtension, OutFormatLCol, OutFormatRCol, InFormatCol, uncompr_f>(
               p_InDataLCol,
               reinterpret_cast<const column<uncompr_f>*>(col),
               outCountEstimate
             );
          }
          case 1: // STATICBP 32
          {
             //return natural_equi_join_t<VectorExtension, DataStructure, OutFormatLCol, OutFormatRCol, InFormatCol, DEFAULT_STATIC_VBP_F(VectorExtension, 32)>::apply(
             return join<VectorExtension, OutFormatLCol, OutFormatRCol, InFormatCol, DEFAULT_STATIC_VBP_F(VectorExtension, 32)>(
               p_InDataLCol,
               reinterpret_cast<const column<DEFAULT_STATIC_VBP_F(VectorExtension, 32)>*>(col),
               outCountEstimate
             );
          }
          case 2: // DYNAMICBP
          {
             //return natural_equi_join_t<VectorExtension, DataStructure, OutFormatLCol, OutFormatRCol, InFormatCol, DEFAULT_DYNAMIC_VBP_F(VectorExtension)>::apply(
             return join<VectorExtension, OutFormatLCol, OutFormatRCol, InFormatCol, DEFAULT_DYNAMIC_VBP_F(VectorExtension)>(
               p_InDataLCol,
               reinterpret_cast<const column<DEFAULT_DYNAMIC_VBP_F(VectorExtension)>*>(col),
               outCountEstimate
             );
          }
        }
      }
};




}
#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_NATURAL_EQUI_COMPR_H
