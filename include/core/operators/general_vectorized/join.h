//
// Created by jpietrzyk on 28.05.19.
//

#ifndef MORPHSTORE_JOIN_H
#define MORPHSTORE_JOIN_H

#include <core/utils/preprocessor.h>
#include <core/storage/column.h>
#include <core/morphing/format.h>

#include <vector/general_vector.h>
#include <vector/primitives/io.h>
#include <vector/primitives/create.h>
#include <vector/primitives/calc.h>

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace morphstore {


   template<
      class VectorExtension,
      class DataStructure
   >
   struct semi_join_build_batch {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
         base_t *& p_InBuildDataPtr,
         size_t const p_Count,
         DataStructure & hs
      ) {
         using namespace vector;
         auto state = hs.template get_lookup_insert_strategy_state< VectorExtension >();
         for(size_t i = 0; i < p_Count; ++i) {
            hs.template insert<VectorExtension>(
               load<VectorExtension, iov::ALIGNED, vector_size_bit::value>( p_InBuildDataPtr ),
               state );
            p_InBuildDataPtr += vector_element_count::value;
         }
      }
   };

   template<
      class VectorExtension,
      class DataStructure
   >
   struct semi_join_probe_batch {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static size_t
      apply(
         base_t *& p_InProbeDataPtr,
         size_t const p_Count,
         base_t * p_OutPosCol,
         DataStructure & hs
      ) {
         using namespace vector;
         auto state = hs.template get_lookup_insert_strategy_state< VectorExtension >();
         size_t resultCount = 0;

         vector_t positionVector = set_sequence<VectorExtension, vector_base_t_granularity::value>(0,1);
         vector_t const incrementVector = set1<VectorExtension, vector_base_t_granularity::value>( vector_element_count::value );
         vector_mask_t lookupResultMask;
         uint8_t hitResultCount;
         for( size_t i = 0; i < p_Count; ++i ) {
            std::tie( lookupResultMask, hitResultCount ) =
               hs.template lookup<VectorExtension>(
                  load<VectorExtension, iov::ALIGNED, vector_size_bit::value>( p_InProbeDataPtr ),
                  state
               );
            compressstore<VectorExtension, iov::UNALIGNED, vector_size_bit::value>(
               p_OutPosCol, positionVector, lookupResultMask);
            p_OutPosCol += hitResultCount;
            resultCount += hitResultCount;
            positionVector = add< VectorExtension >::apply( positionVector, incrementVector );
            p_InProbeDataPtr += vector_element_count::value;
            hitResultCount = 0;
            lookupResultMask = 0;
         }
         return resultCount;
      }
   };


   template<
      class Format,
      class VectorExtension,
      class DataStructure
   >
   struct semi_join {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

      MSV_CXX_ATTRIBUTE_FORCE_INLINE static
      const column<Format> *
      apply(
         column< Format > const * const p_InDataLCol,
         column< Format > const * const p_InDataRCol
      ) {
         using namespace vector;

         const size_t inBuildDataCount = p_InDataLCol->get_count_values();
         const size_t inProbeDataCount = p_InDataRCol->get_count_values();
         base_t * inBuildDataPtr = p_InDataLCol->get_data( );
         base_t * inProbeDataPtr = p_InDataRCol->get_data( );
         DataStructure hs( inBuildDataCount );
         auto outPosCol = new column<Format>(
            (inProbeDataCount * sizeof(uint64_t))
         );
         base_t * outPtr = outPosCol->get_data( );

         size_t const buildVectorCount = inBuildDataCount / vector_element_count::value;
         size_t const buildRemainderCount = inBuildDataCount % vector_element_count::value;
         size_t const probeVectorCount = inProbeDataCount / vector_element_count::value;
         size_t const probeRemainderCount = inProbeDataCount % vector_element_count::value;

         semi_join_build_batch<VectorExtension, DataStructure >::apply( inBuildDataPtr, buildVectorCount, hs );
         semi_join_build_batch<scalar<v64<base_t>>, DataStructure>::apply( inBuildDataPtr, buildRemainderCount, hs );
         size_t resultCount =
            semi_join_probe_batch<VectorExtension, DataStructure>::apply(
               inProbeDataPtr, probeVectorCount, outPtr, hs
            );
         resultCount +=
            semi_join_probe_batch<scalar<v64<base_t>>, DataStructure>::apply(
               inProbeDataPtr, probeRemainderCount, outPtr, hs
            );
         outPosCol->set_meta_data(resultCount, resultCount * sizeof(uint64_t));

         return outPosCol;
      }

   };


}
#endif //MORPHSTORE_JOIN_H
