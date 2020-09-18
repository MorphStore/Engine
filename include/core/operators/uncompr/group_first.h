//
// Created by jpietrzyk on 28.05.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_UNCOMPR_GROUP_FIRST_H
#define MORPHSTORE_CORE_OPERATORS_UNCOMPR_GROUP_FIRST_H

#include <core/utils/preprocessor.h>
#include <core/storage/column.h>
#include <core/morphing/format.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <vector/datastructures/hash_based/hash_utils.h>
#include <vector/datastructures/hash_based/hash_map.h>
#include <vector/datastructures/hash_based/hash_binary_key_map.h>
#include <vector/datastructures/hash_based/strategies/linear_probing.h>

#include <vector/complex/hash.h>

#include <core/operators/interfaces/group_first.h>

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace morphstore {
   template<
      class VectorExtension,
      class DataStructure
   >
   struct group_first_batch_t {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static size_t apply(
         base_t *& p_InDataPtr,
         base_t *& p_OutGroupIdPtr,
         base_t *& p_OutGroupExtPtr,
         size_t const p_Count,
         base_t & p_InPositionIn,
         base_t & p_GroupIdIn,
         DataStructure & hs
      ) {
         using namespace vectorlib;
         base_t groupId = 0;
         size_t result = 0;
         auto state = hs.template get_lookup_insert_strategy_state< VectorExtension >();
         vector_t groupIdVector, groupExtVector;
         vector_mask_t activeGroupExtMask;
         uint8_t activeGroupExtCount;
         for(size_t i = 0; i < p_Count; ++i) {
            std::tie( groupIdVector, groupExtVector, activeGroupExtMask, activeGroupExtCount ) =
               hs.template insert_and_lookup<VectorExtension>(
                  load<VectorExtension, iov::ALIGNED, vector_size_bit::value>( p_InDataPtr ),
                  p_InPositionIn,
                  p_GroupIdIn,
                  state
               );

            store<VectorExtension, iov::ALIGNED, vector_size_bit::value>(
               p_OutGroupIdPtr,
               groupIdVector
            );
            compressstore<VectorExtension, iov::UNALIGNED, vector_size_bit::value>(
               p_OutGroupExtPtr, groupExtVector, activeGroupExtMask);
            p_InDataPtr += vector_element_count::value;
            p_OutGroupIdPtr += vector_element_count::value;
            p_OutGroupExtPtr += activeGroupExtCount;
            result += activeGroupExtCount;
         }
         return result;
      }
   };

   template<class VectorExtension>
   struct group_first_t<
      VectorExtension,
      uncompr_f,
      uncompr_f,
      uncompr_f
   > {
      using DataStructure = hash_map<
         VectorExtension,
         multiply_mod_hash,
         size_policy_hash::EXPONENTIAL,
         scalar_key_vectorized_linear_search,
         60
      >;
       
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

      static
      const std::tuple<
         const column<uncompr_f> *,
         const column<uncompr_f> *
      >
      apply(
         column<uncompr_f> const * const  p_InDataCol,
         size_t const outCountEstimate = 0
      ) {
         using namespace vectorlib;

         const size_t inDataCount = p_InDataCol->get_count_values();
         const size_t inDataSize = p_InDataCol->get_size_used_byte();

         const size_t outCount = bool(outCountEstimate) ? (outCountEstimate): inDataCount;
         auto outGrCol = new column<uncompr_f>(inDataSize);
         auto outExtCol = new column<uncompr_f>( outCount * sizeof( uint64_t ) );

         base_t * inDataPtr = p_InDataCol->get_data( );
         base_t * outGr = outGrCol->get_data();
         base_t * outExt = outExtCol->get_data();
//         base_t * const initOutExt = outExt;

         DataStructure hs( inDataCount );
         size_t const dataVectorCount = inDataCount / vector_element_count::value;
         size_t const dataRemainderCount = inDataCount % vector_element_count::value;

         base_t currentGrId = 0;
         base_t currentPos = 0;

         size_t resultCount = group_first_batch_t<VectorExtension, DataStructure>::apply(
            inDataPtr,
            outGr,
            outExt,
            dataVectorCount,
            currentPos,
            currentGrId,
            hs
         );
         resultCount += group_first_batch_t<scalar<scalar_vector_view<base_t>>, DataStructure>::apply(
            inDataPtr,
            outGr,
            outExt,
            dataRemainderCount,
            currentPos,
            currentGrId,
            hs
         );

         outGrCol->set_meta_data(inDataCount, inDataCount * sizeof(uint64_t));
         outExtCol->set_meta_data(resultCount, resultCount * sizeof(uint64_t));

         return std::make_tuple(outGrCol, outExtCol);
      }
   };
   
}
#endif //MORPHSTORE_CORE_OPERATORS_UNCOMPR_GROUP_FIRST_H
