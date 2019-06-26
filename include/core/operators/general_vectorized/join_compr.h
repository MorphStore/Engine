//
// Created by jpietrzyk on 28.05.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_COMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_COMPR_H


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
   struct semi_join_build_processing_unit_wit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         DataStructure & m_Ds;
         typename DataStructure::template strategy_state< VectorExtension > m_StrategyState;
         state_t(
            DataStructure & p_Ds
         ):
            m_Ds{ p_Ds },
            m_StrategyState{ p_Ds.template get_lookup_insert_strategy_state()} {}
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void apply( vector_t & const p_DataVector, state_t & p_State ) {
         p_State.m_Ds.template insert<VectorExtension>(
            p_DataVector,
            state
         );
      }
   };

   template<
      class VectorExtension,
      class DataStructure,
      class OutFormatCol
   >
   struct semi_join_probe_processing_unit_wit {
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
            base_t p_Pos
         ):
            m_Ds{ p_Ds },
            m_Pos{ set_sequence< VectorExtension, vector_base_t_granularity::value >( p_Pos, 1 ) },
            m_Inc{ set1< VectorExtension, vector_base_t_granularity::value >( vector_element_count::value ) },
            m_WitOutData{p_OutPtr},
            m_StrategyState{ p_Ds.template get_lookup_insert_strategy_state()} {}
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void apply( vector_t & const p_DataVector, state_t & p_State ) {
         vector_mask_t lookupResultMask;
         uint8_t hitResultCount;
         std::tie( lookupResultMask, hitResultCount ) =
            p_State.m_Ds.template lookup<VectorExtension>(
               load<VectorExtension, iov::ALIGNED, vector_size_bit::value>( p_InProbeDataPtr ),
               state
            );
         p_State.m_WitGroupExtents.write(p_State.m_Pos, lookupResultMask, hitResultCount);
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
   struct semi_join_t {
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


         uint8_t       *         outPtr                     = outGroupExtentsCol->get_data();
         uint8_t const * const   startOutDataPtr            = outPtr;


         //Build Part Starts Here
            typename semi_join_build_processing_unit_wit<
               VectorExtension,
               DataStructure
            >::state_t witBuildComprState(
               ds
            );
            decompress_and_process_batch<
               VectorExtension,
               InFormatLCol,
               semi_join_build_processing_unit_wit,
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
                  semi_join_build_processing_unit_wit,
                  DataStructure
               >::apply(
                  inBuildDataPtr, inBuildDataSizeUncomprVecByte, witBuildComprState
               );
               const size_t inBuildSizeScalarRemainderByte = inBuildSizeRestByte % vector_size_byte::value;
               if(inBuildSizeScalarRemainderByte) {
                  typename semi_join_build_processing_unit_wit<
                     scalar<v64<uint64_t>>,
                     DataStructure
                  >::state_t witBuildUncomprState(
                     ds
                  );
                  decompress_and_process_batch<
                     scalar<v64<uint64_t>>,
                     uncompr_f,
                     semi_join_build_processing_unit_wit,
                     DataStructure
                  >::apply(
                     inBuildDataPtr, inBuildSizeScalarRemainderByte, witBuildUncomprState
                  );
               }
            }
         //Build Part Ends Here

         //Probe Part Starts Here
            typename semi_join_probe_processing_unit_wit<
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
               semi_join_probe_processing_unit_wit,
               DataStructure
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
                  semi_join_probe_processing_unit_wit,
                  DataStructure
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
                  typename semi_join_probe_processing_unit_wit<
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
                     semi_join_probe_processing_unit_wit,
                     DataStructure
                  >::apply(
                     inProbeDataPtr, inProbeSizeScalarRemainderByte, witProbeUncomprState
                  );

                  std::tie(
                     std::ignore, std::ignore, outPtr
                  ) = witProbeUncomprState.done();
                  outCountLog = witProbeComprState.m_WitOutData.get_count_values();
               }
            }
         //Probe Part Ends Here
         outPosCol->set_meta_data(
            outCountLog,  outPtr - startOutDataPtr, outSizeComprByte
         );
         retrun outPosCol;
      }
   };



/*   template<
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
         base_t *& p_OutPosCol,
      base_t const p_InPositionIn,
         DataStructure & hs
      ) {
         using namespace vector;
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
   };*/

   template<
      class VectorExtension,
      class DataStructure
   >
   struct semi_join_t<VectorExtension,DataStructure,uncompr_f, uncompr_f, uncompr_f> {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      static
      const column<uncompr_f> *
      apply(
         column< uncompr_f > const * const p_InDataLCol,
         column< uncompr_f > const * const p_InDataRCol,
         size_t const outCountEstimate MSV_CXX_ATTRIBUTE_PPUNUSED = 0
      ) {
         using namespace vector;

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
         using namespace vector;
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
         using namespace vector;
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
   struct join_t<VectorExtension,DataStructure,uncompr_f, uncompr_f, uncompr_f, uncompr_f> {
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
         using namespace vector;

         const size_t inBuildDataCount = p_InDataLCol->get_count_values();
         const size_t inProbeDataCount = p_InDataRCol->get_count_values();

         const size_t outCount = bool(outCountEstimate)
                                 // use given estimate
                                 ? (outCountEstimate)
                                 // use pessimistic estimate
                                 : (inBuildDataCount * inProbeDataCount);

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
#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_JOIN_COMPR_H
