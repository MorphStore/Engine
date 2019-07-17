//
// Created by jpietrzyk on 28.05.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_GROUP_BINARY_COMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_GROUP_BINARYCOMPR_H

#include <core/utils/preprocessor.h>
#include <core/storage/column.h>
#include <core/morphing/format.h>

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
      class DataStructure,
      class OutFormatGroupIds,
      class OutFormatGroupExtents
   >
   struct group_binary_processing_unit_wit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         DataStructure    & m_Ds;
         base_t           * m_GroupIdInPosition;
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
         typename DataStructure::template strategy_state<VectorExtension> m_StrategyState;

         state_t(
            DataStructure  &        p_Ds,
            base_t         * const  p_GroupIdInPosition,
            uint8_t        * const  p_GroupIdsOut,
            uint8_t        * const  p_GroupExtentsOut,
            base_t         &        p_CurrentDataInPosition,
            base_t         &        p_CurrentGroupId
         ):
            m_Ds{p_Ds},
            m_GroupIdInPosition{ p_GroupIdInPosition },
            m_WitGroupIds{p_GroupIdsOut},
            m_WitGroupExtents{p_GroupExtentsOut},
            m_CurrentDataInPosition{p_CurrentDataInPosition},
            m_CurrentGroupId{p_CurrentGroupId},
            m_StrategyState{p_Ds.template get_lookup_insert_strategy_state< VectorExtension >()}{}
      };

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void apply( vector_t const & p_DataVector, state_t & p_State ) {
         vector_t       groupIdInVector = load<VectorExtension, iov::ALIGNED, vector_size_bit::value>( p_State.m_GroupIdInPosition );
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
               groupIdInVector,
               //@todo: this should be a vector register!
               p_State.m_CurrentDataInPosition,
               p_State.m_CurrentGroupId,
               p_State.m_StrategyState
            );
         p_State.m_WitGroupIds.write(groupIdVector);
         p_State.m_WitGroupExtents.write(groupExtVector, activeGroupExtMask, activeGroupExtCount);
         p_State.m_GroupIdInPosition += vector_element_count::value;
      }
   };

   template<
      class VectorExtension,
      class DataStructure,
      class OutFormatGroupIds,
      class OutFormatGroupExtents,
      class InFormatDataCol
   >
   struct group_binary_wit_t<
      VectorExtension,
      DataStructure,
      OutFormatGroupIds,
      OutFormatGroupExtents,
      uncompr_f,
      InFormatDataCol
   > {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      static
      std::tuple<
         column<OutFormatGroupIds> const *,
         column<OutFormatGroupExtents> const *
      > const
      apply(
         column<uncompr_f> const * const p_InGrCol,
         column<InFormatDataCol> const * const p_InDataCol,
         size_t const outCountGroupExtentsEstimate = 0
      ) {
         uint8_t const *         inData               = p_InDataCol->get_data();
         base_t  const *         inGroupId            = p_InGrCol->get_data();
         uint8_t const * const   initInData           = inData;
         size_t  const           inDataCountLog       = p_InDataCol->get_count_values();
         size_t  const           inDataSizeComprByte  = p_InDataCol->get_size_compr_byte();
         size_t  const           inDataSizeUsedByte   = p_InDataCol->get_size_used_byte();

         base_t                  currentDataInPosition = 0;
         base_t                  currentGroupId = 0;

         DataStructure ds( inDataCountLog );

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


         typename group_binary_processing_unit_wit<
            VectorExtension,
            DataStructure,
            OutFormatGroupIds,
            OutFormatGroupExtents
         >::state_t witComprState(
            ds,
            inGroupId,
            outGroupIdsPos,
            outGroupExtentsPos,
            currentDataInPosition,
            currentGroupId
         );


         decompress_and_process_batch<
            VectorExtension,
            InFormatData,
            group_binary_processing_unit_wit,
            DataStructure,
            OutFormatGroupIds,
            OutFormatGroupExtents
         >::apply(
            inData, inDataSizeComprByte, witComprState
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
            inData = create_aligned_ptr(inData);
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
               group_binary_processing_unit_wit,
               DataStructure,
               OutFormatGroupIds,
               OutFormatGroupExtents
            >::apply(
               inData, inDataSizeUncomprVecByte, witComprState
            );

            bool outAddedPaddingGroupIds, outAddedPaddingGroupExtents;

            std::tie(
               outSizeGroupIdComprByte, outAddedPaddingGroupIds, outPosGroupId
            ) = witComprState.m_WitGroupIds.done();
            std::tie(
               outSizeGroupExtentsComprByte, outAddedPaddingGroupExtents, outPosGroupExtents
            ) = witComprState.m_WitGroupExtents.done();


            const size_t inSizeScalarRemainderByte = inSizeRestByte % vector_size_byte::value;
            if(inSizeScalarRemainderByte) {
               if(!outAddedPaddingGroupIds)
                  outPosGroupId = create_aligned_ptr(outPosGroupId);
               if(!outAddedPaddingGroupExtents)
                  outPosGroupExtents = create_aligned_ptr(outPosGroupExtents);

               typename group_binary_processing_unit_wit<
                  scalar<v64<uint64_t>>,
                  DataStructure,
                  uncompr_f,
                  uncompr_f
               >::state_t witUncomprState(
                  ds,
                  inGroupId,
                  outPosGroupId,
                  outPosGroupExtents,
                  witComprState.m_CurrentDataInPosition,
                  witComprState.m_CurrentGroupId
               );


               // Processing of the input column's uncompressed scalar rest
               // part using scalar instructions, uncompressed output.
               decompress_and_process_batch<
                  scalar<v64<uint64_t>>,
                  uncompr_f,
                  group_binary_processing_unit_wit,
                  DataStructure,
                  uncompr_f,
                  uncompr_f
               >::apply(
                  inData, inSizeScalarRemainderByte, witUncomprState
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
}
#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_GROUP_BINARY_COMPR_H
