//
// Created by jpietrzyk on 28.05.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_UNCOMPR_H


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
    using namespace vectorlib;

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
         using namespace vectorlib;
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
         base_t *& p_OutPosCol,
         base_t const p_InPositionIn,
         DataStructure & hs
      ) {
         using namespace vectorlib;
         auto state = hs.template get_lookup_insert_strategy_state< VectorExtension >();
         size_t resultCount = 0;

         vector_t positionVector = set_sequence<VectorExtension, vector_base_t_granularity::value>(p_InPositionIn,1);
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
      class VectorExtension,
      class DataStructure
   >
   struct semi_equi_join_t<VectorExtension,DataStructure,uncompr_f, uncompr_f, uncompr_f> {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      static
      const column<uncompr_f> *
      apply(
         column< uncompr_f > const * const p_InDataLCol,
         column< uncompr_f > const * const p_InDataRCol,
         size_t const outCountEstimate MSV_CXX_ATTRIBUTE_PPUNUSED = 0
      ) {
         using namespace vectorlib;

         const size_t inBuildDataCount = p_InDataLCol->get_count_values();
         const size_t inProbeDataCount = p_InDataRCol->get_count_values();
         base_t * inBuildDataPtr = p_InDataLCol->get_data( );
         base_t * inProbeDataPtr = p_InDataRCol->get_data( );
         base_t * const startProbeDataPrt = inProbeDataPtr;
         DataStructure hs( inBuildDataCount );
         auto outPosCol = new column<uncompr_f>(
            (inProbeDataCount * sizeof(uint64_t))
         );
         base_t * outPtr = outPosCol->get_data( );

         size_t const buildVectorCount = inBuildDataCount / vector_element_count::value;
         size_t const buildRemainderCount = inBuildDataCount % vector_element_count::value;
         size_t const probeVectorCount = inProbeDataCount / vector_element_count::value;
         size_t const probeRemainderCount = inProbeDataCount % vector_element_count::value;

         semi_join_build_batch<VectorExtension, DataStructure >::apply( inBuildDataPtr, buildVectorCount, hs );
         semi_join_build_batch<scalar<scalar_vector_view<base_t>>, DataStructure>::apply( inBuildDataPtr, buildRemainderCount, hs );
         size_t resultCount =
            semi_join_probe_batch<VectorExtension, DataStructure>::apply(
               inProbeDataPtr, probeVectorCount, outPtr, 0, hs
            );
         resultCount +=
            semi_join_probe_batch<scalar<scalar_vector_view<base_t>>, DataStructure>::apply(
               inProbeDataPtr, probeRemainderCount, outPtr, (inProbeDataPtr-startProbeDataPrt), hs
            );
         outPosCol->set_meta_data(resultCount, resultCount * sizeof(uint64_t));

         return outPosCol;
      }

   };


   template<
      class VectorExtension,
      class DataStructure
   >
   struct equi_join_build_batch {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
         base_t *& p_InBuildDataPtr,
         size_t const p_Count,
         base_t const p_InPositionIn,
         DataStructure & hs
      ) {
         using namespace vectorlib;
         vector_t positionVector = set_sequence<VectorExtension, vector_base_t_granularity::value>(p_InPositionIn,1);
         vector_t const incrementVector = set1<VectorExtension, vector_base_t_granularity::value>( vector_element_count::value);
         auto state = hs.template get_lookup_insert_strategy_state< VectorExtension >();
         for(size_t i = 0; i < p_Count; ++i) {
            hs.template insert<VectorExtension>(
               load<VectorExtension, iov::ALIGNED, vector_size_bit::value>( p_InBuildDataPtr ),
               positionVector,
               state );
            p_InBuildDataPtr += vector_element_count::value;
            positionVector = add< VectorExtension >::apply( positionVector, incrementVector );
         }
      }
   };

   template<
      class VectorExtension,
      class DataStructure
   >
   struct equi_join_probe_batch {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static size_t
      apply(
         base_t *& p_InProbeDataPtr,
         size_t const p_Count,
         base_t *& p_OutPosLCol,
         base_t *& p_OutPosRCol,
         base_t const p_InPositionIn,
         DataStructure & hs
      ) {
         using namespace vectorlib;
         auto state = hs.template get_lookup_insert_strategy_state< VectorExtension >();
         size_t resultCount = 0;

         vector_t positionVector = set_sequence<VectorExtension, vector_base_t_granularity::value>(p_InPositionIn,1);
         vector_t const incrementVector = set1<VectorExtension, vector_base_t_granularity::value>( vector_element_count::value );
         vector_t lookupResultValuesVector;
         vector_mask_t lookupResultMask;
         uint8_t hitResultCount;

         for( size_t i = 0; i < p_Count; ++i ) {
            std::tie( lookupResultValuesVector, lookupResultMask, hitResultCount ) =
               hs.template lookup<VectorExtension>(
                  load<VectorExtension, iov::ALIGNED, vector_size_bit::value>( p_InProbeDataPtr ),
                  state
               );
            compressstore<VectorExtension, iov::UNALIGNED, vector_size_bit::value>(
               p_OutPosLCol, lookupResultValuesVector, lookupResultMask);
            compressstore<VectorExtension, iov::UNALIGNED, vector_size_bit::value>(
               p_OutPosRCol, positionVector, lookupResultMask);
            p_OutPosRCol += hitResultCount;
            p_OutPosLCol += hitResultCount;
            resultCount += hitResultCount;
            positionVector = add< VectorExtension >::apply( positionVector, incrementVector );
            p_InProbeDataPtr += vector_element_count::value;
            hitResultCount = 0;
            lookupResultMask = 0;
         }

         return resultCount;
      }
   };

   //<VectorExtension,DataStructure,uncompr_f, uncompr_f, uncompr_f, uncompr_f>
   template<
      class VectorExtension,
      class DataStructure
   >
   struct natural_equi_join_t<VectorExtension,DataStructure,uncompr_f, uncompr_f, uncompr_f, uncompr_f> {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      static
      const std::tuple<
         const column<uncompr_f> *,
         const column<uncompr_f> *
      >
      apply(
         column< uncompr_f > const * const p_InDataLCol,
         column< uncompr_f > const * const p_InDataRCol,
         size_t const outCountEstimate = 0
      ) {
         using namespace vectorlib;

         const size_t inBuildDataCount = p_InDataLCol->get_count_values();
         const size_t inProbeDataCount = p_InDataRCol->get_count_values();

         const size_t outCount = bool(outCountEstimate)
                             // use given estimate
                             ? (outCountEstimate)
                             // use pessimistic estimate
                             : (inProbeDataCount);

         base_t * inBuildDataPtr = p_InDataLCol->get_data( );
         base_t * inProbeDataPtr = p_InDataRCol->get_data( );
         base_t * const startBuildDataPtr = inBuildDataPtr;
         base_t * const startProbeDataPtr = inProbeDataPtr;
         DataStructure hs( inBuildDataCount );
         auto outPosLCol = new column<uncompr_f>(
            (outCount * sizeof(uint64_t))
         );
         auto outPosRCol = new column<uncompr_f>(
            (outCount * sizeof(uint64_t))
         );
         base_t * outLPtr = outPosLCol->get_data( );
         base_t * outRPtr = outPosRCol->get_data( );

         size_t const buildVectorCount = inBuildDataCount / vector_element_count::value;
         size_t const buildRemainderCount = inBuildDataCount % vector_element_count::value;
         size_t const probeVectorCount = inProbeDataCount / vector_element_count::value;
         size_t const probeRemainderCount = inProbeDataCount % vector_element_count::value;

         equi_join_build_batch<VectorExtension, DataStructure >::apply( inBuildDataPtr, buildVectorCount, 0, hs );
         equi_join_build_batch<scalar<scalar_vector_view<base_t>>, DataStructure>::apply(
            inBuildDataPtr, buildRemainderCount, (inBuildDataPtr-startBuildDataPtr), hs
         );
         size_t resultCount =
            equi_join_probe_batch<VectorExtension, DataStructure>::apply(
               inProbeDataPtr, probeVectorCount, outLPtr, outRPtr, 0, hs
         );
         resultCount +=
            equi_join_probe_batch<scalar<scalar_vector_view<base_t>>, DataStructure>::apply(
               inProbeDataPtr, probeRemainderCount, outLPtr, outRPtr, (inProbeDataPtr-startProbeDataPtr), hs
         );
         outPosLCol->set_meta_data(resultCount, resultCount * sizeof(uint64_t));
         outPosRCol->set_meta_data(resultCount, resultCount * sizeof(uint64_t));

         return std::make_tuple(outPosLCol, outPosRCol);
      }
   };

}
#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_UNCOMPR_H
