//
// Created by jpietrzyk on 28.05.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_GROUP_COMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_GROUP_COMPR_H

#include <core/utils/preprocessor.h>
#include <core/storage/column.h>
#include <core/morphing/format.h>
#include <core/morphing/write_iterator.h>

#include <vector/general_vector_extension.h>
#include <vector/vector_primitives.h>

#include <vector/datastructures/hash_based/hash_utils.h>
#include <vector/datastructures/hash_based/hash_map.h>
#include <vector/datastructures/hash_based/hash_binary_key_map.h>
#include <vector/datastructures/hash_based/strategies/linear_probing.h>

#include <vector/complex/hash.h>

#include <core/operators/interfaces/group.h>

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace morphstore {
   using namespace vectorlib;

   template<
      class VectorExtension,
      class OutFormatGroupIds,
      class OutFormatGroupExtents,
      class DataStructure
   >
   struct group_processing_unit_wit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         DataStructure    & m_Ds;
         nonselective_write_iterator<
            VectorExtension,
            OutFormatGroupIds
         >                 m_WitGroupIds;
         selective_write_iterator<
            VectorExtension,
            OutFormatGroupExtents
         >                 m_WitGroupExtents;
         base_t          & m_CurrentDataInPosition;
         base_t          & m_CurrentGroupId;
//         size_t            m_ResultCount;
         typename DataStructure::template strategy_state<VectorExtension> m_StrategyState;

         state_t(
            DataStructure  &        p_Ds,
            uint8_t        * const  p_GroupIdsOut,
            uint8_t        * const  p_GroupExtentsOut,
            base_t         &        p_CurrentDataInPosition,
            base_t         &        p_CurrentGroupId
         ):
            m_Ds{p_Ds},
            m_WitGroupIds{p_GroupIdsOut},
            m_WitGroupExtents{p_GroupExtentsOut},
            m_CurrentDataInPosition{p_CurrentDataInPosition},
            m_CurrentGroupId{p_CurrentGroupId},
            m_StrategyState{p_Ds.template get_lookup_insert_strategy_state< VectorExtension >()}{}
      };

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void apply( vector_t const & p_DataVector, state_t & p_State ) {
         vector_t       groupIdVector, groupExtVector;
         vector_mask_t  activeGroupExtMask;
         uint8_t        activeGroupExtCount;
         std::tie(
            groupIdVector,
            groupExtVector,
            activeGroupExtMask,
            activeGroupExtCount
         ) =
            p_State.m_Ds.template insert_and_lookup<VectorExtension>(
               p_DataVector,
               //@todo: this should be a vector register!
               p_State.m_CurrentDataInPosition,
               p_State.m_CurrentGroupId,
               p_State.m_StrategyState
            );
         p_State.m_WitGroupIds.write(groupIdVector);
         p_State.m_WitGroupExtents.write(groupExtVector, activeGroupExtMask, activeGroupExtCount);
      }
   };

   template<
      class VectorExtension,
      class OutFormatGroupIds,
      class OutFormatGroupExtents,
      class InFormatData,
      class DataStructure
   >
   struct group_wit_t {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      static
      std::tuple<
         column<OutFormatGroupIds> const *,
         column<OutFormatGroupExtents> const *
      > const
      apply(
         column<InFormatData> const * const p_InDataCol,
         size_t const outCountGroupExtentsEstimate = 0
      ) {
         uint8_t const *         inData               = p_InDataCol->get_data();
         uint8_t const * const   initInData           = inData;
         size_t  const           inDataCountLog       = p_InDataCol->get_count_values();
         size_t  const           inDataSizeComprByte  = p_InDataCol->get_size_compr_byte();
         size_t  const           inDataSizeUsedByte   = p_InDataCol->get_size_used_byte();

         base_t                  currentDataInPosition = 0;
         base_t                  currentGroupId = 0;

         DataStructure ds( inDataCountLog );

//         const size_t inCountLogRest = convert_size<uint8_t, uint64_t>(
//            inDataSizeUsedByte - (inDataRest8 - inData)
//         );

         auto outGroupExtentsCol = new column<OutFormatGroupExtents>(
            bool(outCountGroupExtentsEstimate)
            // use given estimate
            ? get_size_max_byte_any_len<OutFormatGroupExtents>(outCountGroupExtentsEstimate)
            // use pessimistic estimate
            : get_size_max_byte_any_len<OutFormatGroupExtents>(inDataCountLog)
         );

         auto outGroupIdsCol = new column<OutFormatGroupIds>(
            get_size_max_byte_any_len<OutFormatGroupIds>(inDataCountLog)
         );

         uint8_t       *         outGroupExtentsPos      = outGroupExtentsCol->get_data();
         uint8_t const * const   initOutGroupExtentsPos  = outGroupExtentsPos;
         uint8_t       *         outGroupIdsPos          = outGroupIdsCol->get_data();
         uint8_t const * const   initOutGroupIdsPos      = outGroupIdsPos;



         typename group_processing_unit_wit<
            VectorExtension,
            OutFormatGroupIds,
            OutFormatGroupExtents,
            DataStructure
         >::state_t witComprState(
            ds,
            outGroupIdsPos,
            outGroupExtentsPos,
            currentDataInPosition,
            currentGroupId
         );

         decompress_and_process_batch<
            VectorExtension,
            InFormatData,
            group_processing_unit_wit,
            OutFormatGroupIds,
            OutFormatGroupExtents,
            DataStructure
         >::apply(
            inData, p_InDataCol->get_count_values_compr(), witComprState
         );

         size_t outSizeGroupIdComprByte, outSizeGroupExtentsComprByte;
         uint8_t * outPosGroupId;
         uint8_t * outPosGroupExtents;

         size_t groupExtentsCount;
         if(inDataSizeComprByte == inDataSizeUsedByte) {
            // If the input column has no uncompressed rest part, we are done.
            std::tie(
               outSizeGroupIdComprByte, std::ignore, outPosGroupId
            ) = witComprState.m_WitGroupIds.done();
            std::tie(
               outSizeGroupExtentsComprByte, std::ignore, outPosGroupExtents
            ) = witComprState.m_WitGroupExtents.done();
            groupExtentsCount = witComprState.m_CurrentGroupId;
         } else {
            inData = p_InDataCol->get_data_uncompr_start();
            // The size of the input column's uncompressed rest part.
            const size_t inSizeRestByte = initInData + inDataSizeUsedByte - inData;

            // Vectorized processing of the input column's uncompressed rest
            // part using the specified vector extension, compressed output.
            const size_t inDataSizeUncomprVecByte = round_down_to_multiple(
               inSizeRestByte, vector_size_byte::value
            );
            decompress_and_process_batch<
               VectorExtension,
               uncompr_f,
               group_processing_unit_wit,
               OutFormatGroupIds,
               OutFormatGroupExtents,
               DataStructure
            >::apply(
               inData,
               convert_size<uint8_t, uint64_t>(inDataSizeUncomprVecByte),
               witComprState
            );

            uint8_t * outAppendUncomprGroupIds;
            uint8_t * outAppendUncomprGroupExtents;

            std::tie(
               outSizeGroupIdComprByte, outAppendUncomprGroupIds, outPosGroupId
            ) = witComprState.m_WitGroupIds.done();
            std::tie(
               outSizeGroupExtentsComprByte, outAppendUncomprGroupExtents, outPosGroupExtents
            ) = witComprState.m_WitGroupExtents.done();


            const size_t inSizeScalarRemainderByte = inSizeRestByte % vector_size_byte::value;
            if(inSizeScalarRemainderByte) {
               typename group_processing_unit_wit<
                  scalar<v64<uint64_t>>,
                  uncompr_f,
                  uncompr_f,
                  DataStructure
               >::state_t witUncomprState(
                  ds,
                  outAppendUncomprGroupIds,
                  outAppendUncomprGroupExtents,
                  witComprState.m_CurrentDataInPosition,
                  witComprState.m_CurrentGroupId
               );


               // Processing of the input column's uncompressed scalar rest
               // part using scalar instructions, uncompressed output.
               decompress_and_process_batch<
                  scalar<v64<uint64_t>>,
                  uncompr_f,
                  group_processing_unit_wit,
                  uncompr_f,
                  uncompr_f,
                  DataStructure
               >::apply(
                  inData,
                  convert_size<uint8_t, uint64_t>(inSizeScalarRemainderByte),
                  witUncomprState
               );

               // Finish the output column's uncompressed rest part.
               std::tie(
                  std::ignore, std::ignore, outPosGroupId
               ) = witUncomprState.m_WitGroupIds.done();
               std::tie(
                  std::ignore, std::ignore, outPosGroupExtents
               ) = witUncomprState.m_WitGroupExtents.done();

               groupExtentsCount = witUncomprState.m_CurrentGroupId;
            } else {
               groupExtentsCount = witComprState.m_CurrentGroupId;
            }
         }

         outGroupIdsCol->set_meta_data(
            inDataCountLog, outPosGroupId - initOutGroupIdsPos, outSizeGroupIdComprByte
         );
         outGroupExtentsCol->set_meta_data(
            groupExtentsCount, outPosGroupExtents - initOutGroupExtentsPos, outSizeGroupExtentsComprByte
         );

         return std::make_tuple(outGroupIdsCol, outGroupExtentsCol);
      }
   };









   template<class VectorExtension, class t_out_gr_f, class t_out_ext_f, class t_in_data_f>
   static
   const std::tuple<
      const column<t_out_gr_f> *,
      const column<t_out_ext_f> *
   > group_vec(
      column<t_in_data_f> const * const  p_InDataCol,
      size_t const outCountEstimate = 0
   ) {
      return group_wit_t<VectorExtension,
         t_out_gr_f,
         t_out_ext_f,
         t_in_data_f,
         hash_map<
            VectorExtension,
            multiply_mod_hash,
            size_policy_hash::EXPONENTIAL,
            scalar_key_vectorized_linear_search,
            60>
      >::apply(p_InDataCol,outCountEstimate);
   }

}
#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_GROUP_COMPR_H
