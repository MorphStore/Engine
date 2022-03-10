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
 * @file join_natural_equi_compr.h
 * @brief The template-based implementation of the natural-equi-join-operator.
 * @details
 * A Join-operator in MorphStore is implemented as a two-phase operator consisting of a 'Build' and a 'Probe' phase.
 * Within the 'Build' phase, the left column is consumed and its elements (keys) are inserted into an underlying
 * data structure (ds) along with their positions.
 * Example:
 * - p_in_L_pos_column:          [ 1, 2, 4, 5 ]
 * - -> internal data_structure: ( [ 4, 1, 5, 2 ] [ 2, 0, 3, 1 ] ).
 *
 * This ds is then used within the 'Probe' phase to check (lookup), whether the elements from the right column have
 * matches.
 * Example:
 * - p_in_L_pos_column:          [ 3, 4, 5, 4 ]
 * - internal data_structure:    ( [ 4, 1, 5, 2 ] [ 2, 0, 3, 1 ] )
 * - p_out_L_pos_column:         [ 2, 2, 3 ]
 * - p_out_R_pos_column:         [ 1, 3, 2 ].
 *
 * Consequently the Join-operator as it is has only to provide the functionality to iterate over both columns in two
 * phases, forward the elements to the ds, retrieve the result from the probing phase and create the output.
 * However, the real working horse is the ds. For a vectorized natural equi join we are using a hash-map< T, T >
 * where T must be an arithmetic type.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_OTFLY_DERECOMPR_JOIN_NATURAL_EQUI_COMPR_H
#define MORPHSTORE_CORE_OPERATORS_OTFLY_DERECOMPR_JOIN_NATURAL_EQUI_COMPR_H

#include <core/utils/preprocessor.h>
#include <core/storage/column.h>
#include <core/morphing/format.h>
#include <core/morphing/write_iterator.h>

#include <core/utils/basic_types.h>

#include <core/operators/interfaces/natural_equi_join.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <vector/complex/hash.h>
#include <vector/datastructures/hash_based/strategies/linear_probing.h>
#include <vector/datastructures/hash_based/hash_utils.h>
#include <vector/datastructures/hash_based/hash_map.h>

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace morphstore {
   using namespace vectorlib;

   /**
    * @struct natural_equi_join_build_processing_unit_t
    * @brief Processing unit of the build phase for Natural-Join-operator implementation.
    * @details Processing unit struct which executes a step within the building phase of a natural equi join.
    * @tparam t_vector_extension Used vector extension.
    * @tparam t_data_structure Used internal data structure.
    */
   template<
      class t_vector_extension,
      class t_data_structure
   >
   struct natural_equi_join_build_processing_unit_t {
      IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
      /**
       * @struct state_t
       * @brief Struct for holding the internal state of the build phase of the natural-equi-join-operator.
       * @var state_t::m_data_structure
       * Member 'm_data_structure' contains a reference to the underlying data structure (e.g. hash-map).
       * @var state_t::m_build_pos
       * Member 'm_build_pos' contains a vector register which holds the current positions of the input column
       * (e.g. [ 0, 1, 2, 3 ]).
       * @var state_t::m_inc
       * Member 'm_inc' contains a vector register which holds the number of values which fit into a vector register
       * (e.g. avx2< uint64_t >: [ 4, 4, 4, 4 ]). This is the step width for iterating through the data.
       * @var state_t::m_strategy_state
       * Member 'm_strategy_state' contains a state object to the used build strategy.
       */
      struct state_t {
         t_data_structure & m_data_structure;
         vector_t           m_build_pos;
         vector_t const     m_inc;
         typename t_data_structure::template strategy_state< t_vector_extension >
                            m_strategy_state;
         /**
          * @brief Constructs a state_t.
          * @param p_data_structure Reference to an instance of the underlying data structure.
          * @param p_build_pos Current position within the build column.
          */
         state_t(
            t_data_structure &    p_data_structure,
            base_t          const p_build_pos
         ):
            m_data_structure{ p_data_structure },
            m_build_pos{ set_sequence< t_vector_extension, vector_base_t_granularity::value >( p_build_pos, 1 ) },
            m_inc{ set1< t_vector_extension, vector_base_t_granularity::value >( vector_element_count::value ) },
            m_strategy_state{ p_data_structure.template get_lookup_insert_strategy_state< t_vector_extension >( ) } {}
      };

      /**
       * @brief Inserts the values of a given vector register into the internal data structure.
       * @param p_data_vector The vector holding elements which are used as keys for insertion into the internal data
       * structure.
       * @param p_state Current state of the operator.
       */
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void apply( vector_t const & p_data_vector, state_t & p_state ) {
         p_state.m_data_structure.template insert< t_vector_extension >(
            p_data_vector,
            p_state.m_build_pos,
            p_state.m_strategy_state
         );
         p_state.m_build_pos = add< t_vector_extension >::apply( p_state.m_build_pos, p_state.m_inc );
      }
   };

   /**
    * @struct natural_equi_join_probe_processing_unit_t
    * @brief Processing unit of the probe phase for Natural-Join-operator implementation.
    * @details Processing unit struct which executes a step within the probing phase of a natural equi join.
    * @tparam t_vector_extension Used vector extension.
    * @tparam t_data_structure Used internal data structure.
    * @tparam t_out_pos_l_f Compression format for the position list of the left result column.
    * @tparam t_out_pos_r_f Compression format for the position list of the right result column.
    */
   template<
      class t_vector_extension,
      class t_data_structure,
      class t_out_pos_l_f,
      class t_out_pos_r_f
   >
   struct natural_equi_join_probe_processing_unit_t {
      IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
      /**
       * @struct state_t
       * @brief Struct for holding the internal state of the probe phase of the natural-equi-join-operator.
       * @var state_t::m_data_structure
       * Member 'm_data_structure' contains a reference to the underlying data structure (e.g. hash-map).
       * @var state_t::m_probe_pos
       * Member 'm_probe_pos' contains a vector register which holds the current positions of the input column
       * (e.g. [ 0, 1, 2, 3 ]).
       * @var state_t::m_inc
       * Member 'm_inc' contains a vector register which holds the number of values which fit into a vector register
       * (e.g. avx2< uint64_t >: [ 4, 4, 4, 4 ]). This is the step width for iterating through the data.
       * @var state_t::m_strategy_state
       * Member 'm_strategy_state' contains a state object to the used build strategy.
       * @var state_t::m_write_it_out_l_pos
       * Member 'm_write_it_out_l_pos' contains an instance of a selective write iterator for the left result column.
       * @var state_t::m_write_it_out_r_pos
       * Member 'm_write_it_out_r_pos' contains an instance of a selective write iterator for the right result column.
       */
      struct state_t {
         t_data_structure & m_data_structure;
         vector_t          m_probe_pos;
         vector_t const    m_inc;
         typename t_data_structure::template strategy_state< t_vector_extension >
                           m_strategy_state;
         selective_write_iterator<
            t_vector_extension,
            t_out_pos_l_f
         >                 m_write_it_out_l_pos;
         selective_write_iterator<
            t_vector_extension,
            t_out_pos_r_f
         >                 m_write_it_out_r_pos;

         /**
          * @brief Constructs a state_t.
          * @param p_data_structure Reference to an instance of the underlying data structure.
          * @param p_OutLData Pointer to the current position within the left result column.
          * @param p_OutRData Pointer to the current position within the right result column.
          * @param p_ProbePos Current position within the probe column.
          */
         state_t(
            t_data_structure  &        p_data_structure,
            uint8_t        * const  p_OutLData,
            uint8_t        * const  p_OutRData,
            base_t           const  p_ProbePos
         ):
            m_data_structure{ p_data_structure },
            m_probe_pos{ set_sequence< t_vector_extension, vector_base_t_granularity::value >( p_ProbePos, 1 ) },
            m_inc{ set1< t_vector_extension, vector_base_t_granularity::value >( vector_element_count::value ) },
            m_strategy_state{ p_data_structure.template get_lookup_insert_strategy_state< t_vector_extension >()},
            m_write_it_out_l_pos{ p_OutLData },
            m_write_it_out_r_pos{ p_OutRData } {}
      };

      /**
       * @brief Probes the values of a given vector register whether they have a matching element within the internal 
       * data structure.
       * @param p_data_vector The vector holding elements which should be looked up within the internal ds.
       * @param p_state Current state of the operator.
       */
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void apply( vector_t const & p_data_vector, state_t & p_state ) {
         vector_t       lookup_result_values_vector;
         vector_mask_t  lookup_result_mask;
         uint8_t        hit_result_count;
         std::tie( lookup_result_values_vector, lookup_result_mask, hit_result_count ) =
            p_state.m_data_structure.template lookup< t_vector_extension >(
               p_data_vector,
               p_state.m_strategy_state
            );
         p_state.m_write_it_out_l_pos.write( lookup_result_values_vector, lookup_result_mask, hit_result_count );
         p_state.m_write_it_out_r_pos.write( p_state.m_probe_pos, lookup_result_mask, hit_result_count );
         p_state.m_probe_pos = add< t_vector_extension >::apply( p_state.m_probe_pos, p_state.m_inc );
      }
   };


   /**
    * @struct natural_equi_join_t
    * @brief Definition of Natural-Join-operator.
    * @tparam t_vector_extension Used vector extension.
    * @tparam t_out_pos_l_f Compression format for the position list of the left result column.
    * @tparam t_out_pos_r_f Compression format for the position list of the right result column.
    * @tparam t_in_pos_l_f Compression format for the position list of the left input column.
    * @tparam t_in_pos_r_f Compression format for the position list of the right input column.
    */
   template<
      class t_vector_extension,
      class t_out_pos_l_f,
      class t_out_pos_r_f,
      class t_in_pos_l_f,
      class t_in_pos_r_f
   >
   struct natural_equi_join_t {
      IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
      /**
       * @brief Used internal data structure.
       * @details The underlying hash-map is parameterized as followed:
       * - Hashmethod: Multiply-Modulo
       * - Resizepolicy: Exponential
       * - Maximum Load Factor: 60%.
       */
      using t_data_structure =
         vectorlib::hash_map<
            t_vector_extension,
            vectorlib::multiply_mod_hash,
            vectorlib::size_policy_hash::EXPONENTIAL,
            vectorlib::scalar_key_vectorized_linear_search,
            60
         >;
      
   
      /**
       * @brief Implementation of a Natural-Join-operator.
       * @details Joins the two given columns and returns the matching pairs of positions
       * in the form of two position columns of the same length.
       * Example:
       * - p_in_L_pos_column:  [ 22, 44, 11, 33, 55, 77 ]
       * - p_in_R_pos_column:  [ 33, 22, 22, 11 ]
       * - p_out_L_pos_column: [  0,  0,  2,  3 ]
       * - p_out_R_pos_column: [  1,  2,  3,  0 ]
       * Internally, a hash-map is used as an intermediate uncompressed datastructure.
       * The left column is used to build up the map and the right column is probed
       * against the map. Thus the used hash-map is optimized for vectorized processing,
       * it only supports arithmetic types for keys and values. Consequently, the left
       * column must consist of only unique values.
       * @param p_in_L_pos_column The left column to join on, containing only unique data elements.
       * @param p_in_R_pos_column The right column to join on.
       * @param p_out_count_estimate An estimate of the number of data elements in the output column.
       * @return A tuple of two columns containing equally many data elements. The
       * first (second) output column contains positions referring to the left
       * (right) input column. The i-th positions in the two output columns denote
       * one matching pair.
       */
      static
      std::tuple<
         column< t_out_pos_l_f > const *,
         column< t_out_pos_r_f > const *
      > const
      apply(
         column< t_in_pos_l_f > const * const p_in_L_pos_column,
         column< t_in_pos_r_f > const * const p_in_R_pos_column,
         size_t const p_out_count_estimate = 0
      ) {
         uint8_t const *         inBuildDataPtr             = p_in_L_pos_column->get_data();
         uint8_t const * const   startBuildDataPtr          = inBuildDataPtr;
         uint8_t const *         inProbeDataPtr             = p_in_R_pos_column->get_data();
         uint8_t const * const   startProbeDataPtr          = inProbeDataPtr;

         size_t  const           inBuildDataCountLog        = p_in_L_pos_column->get_count_values();
         size_t  const           inProbeDataCountLog        = p_in_R_pos_column->get_count_values();

         size_t  const           inBuildDataSizeComprByte   = p_in_L_pos_column->get_size_compr_byte();
         size_t  const           inProbeDataSizeComprByte   = p_in_R_pos_column->get_size_compr_byte();

         size_t  const           inBuildDataSizeUsedByte    = p_in_L_pos_column->get_size_used_byte();
         size_t  const           inProbeDataSizeUsedByte    = p_in_R_pos_column->get_size_used_byte();

         uint8_t const * const   inProbeDataRest8           = p_in_R_pos_column->get_data_uncompr_start();
         const size_t inProbeCountLogRest                   = convert_size<uint8_t, uint64_t>(
            inProbeDataSizeUsedByte - (inProbeDataRest8 - inProbeDataPtr)
         );
         const size_t inProbeCountLogCompr = inProbeDataCountLog - inProbeCountLogRest;


         uint8_t const * const   inBuildDataRest8           = p_in_L_pos_column->get_data_uncompr_start();
         const size_t inBuildCountLogRest                   = convert_size<uint8_t, uint64_t>(
            inBuildDataSizeUsedByte - (inBuildDataRest8 - inBuildDataPtr)
         );
         const size_t inBuildCountLogCompr = inBuildDataCountLog - inBuildCountLogRest;


         size_t outCountLog;
         t_data_structure ds( inBuildDataCountLog );

         auto outPosLCol = new column<t_out_pos_l_f>(
            bool(p_out_count_estimate)
            // use given estimate
            ? get_size_max_byte_any_len<t_out_pos_l_f>(p_out_count_estimate)
            // use pessimistic estimate
            : get_size_max_byte_any_len<t_out_pos_l_f>(inProbeDataCountLog)
         );
         auto outPosRCol = new column<t_out_pos_r_f>(
            bool(p_out_count_estimate)
            // use given estimate
            ? get_size_max_byte_any_len<t_out_pos_r_f>(p_out_count_estimate)
            // use pessimistic estimate
            : get_size_max_byte_any_len<t_out_pos_r_f>(inProbeDataCountLog)
         );


         uint8_t       *         outLPtr                     = outPosLCol->get_data();
         uint8_t const * const   startOutLDataPtr            = outLPtr;
         uint8_t       *         outRPtr                     = outPosRCol->get_data();
         uint8_t const * const   startOutRDataPtr            = outRPtr;


         //Build Part Starts Here
         typename natural_equi_join_build_processing_unit_t<
            t_vector_extension,
            t_data_structure
         >::state_t witBuildComprState(
            ds,
            0
         );
         decompress_and_process_batch<
            t_vector_extension,
            t_in_pos_l_f,
            natural_equi_join_build_processing_unit_t,
            t_data_structure
         >::apply(
            inBuildDataPtr,
            p_in_L_pos_column->get_count_values_compr(),
            witBuildComprState
         );

         if(inBuildDataSizeComprByte != inBuildDataSizeUsedByte) {
            inBuildDataPtr = inBuildDataRest8;
            size_t const inBuildSizeRestByte = startBuildDataPtr + inBuildDataSizeUsedByte - inBuildDataPtr;

            const size_t inBuildDataSizeUncomprVecByte = round_down_to_multiple(
               inBuildSizeRestByte, vector_size_byte::value
            );

            decompress_and_process_batch<
               t_vector_extension,
               uncompr_f,
               natural_equi_join_build_processing_unit_t,
               t_data_structure
            >::apply(
               inBuildDataPtr,
               convert_size<uint8_t, uint64_t>(inBuildDataSizeUncomprVecByte),
               witBuildComprState
            );
            const size_t inBuildSizeScalarRemainderByte = inBuildSizeRestByte % vector_size_byte::value;
            if(inBuildSizeScalarRemainderByte) {
               typename natural_equi_join_build_processing_unit_t<
                  scalar<v64<uint64_t>>,
                  t_data_structure
               >::state_t witBuildUncomprState(
                  ds,
                  inBuildCountLogCompr + inBuildDataSizeUncomprVecByte / sizeof(base_t)
               );
               decompress_and_process_batch<
                  scalar<v64<uint64_t>>,
                  uncompr_f,
                  natural_equi_join_build_processing_unit_t,
                  t_data_structure
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
         typename natural_equi_join_probe_processing_unit_t<
            t_vector_extension,
            t_data_structure,
            t_out_pos_l_f,
            t_out_pos_r_f
         >::state_t witProbeComprState(
            ds,
            outLPtr,
            outRPtr,
            0
         );
         decompress_and_process_batch<
            t_vector_extension,
            t_in_pos_r_f,
            natural_equi_join_probe_processing_unit_t,
            t_data_structure,
            t_out_pos_l_f,
            t_out_pos_r_f
         >::apply(
            inProbeDataPtr,
            p_in_R_pos_column->get_count_values_compr(),
            witProbeComprState
         );
         size_t outSizeLComprByte;
         size_t outSizeRComprByte;

         if( inProbeDataSizeComprByte == inProbeDataSizeUsedByte ) {
            std::tie(
               outSizeLComprByte, std::ignore, outLPtr
            ) = witProbeComprState.m_write_it_out_l_pos.done();
            std::tie(
               outSizeRComprByte, std::ignore, outRPtr
            ) = witProbeComprState.m_write_it_out_r_pos.done();
            outCountLog = witProbeComprState.m_write_it_out_l_pos.get_count_values();
         } else {
            inProbeDataPtr = inProbeDataRest8;
            size_t const inProbeSizeRestByte = startProbeDataPtr + inProbeDataSizeUsedByte - inProbeDataPtr;
            const size_t inProbeDataSizeUncomprVecByte = round_down_to_multiple(
               inProbeSizeRestByte, vector_size_byte::value
            );
            decompress_and_process_batch<
               t_vector_extension,
               uncompr_f,
               natural_equi_join_probe_processing_unit_t,
               t_data_structure,
               t_out_pos_l_f,
               t_out_pos_r_f
            >::apply(
               inProbeDataPtr,
               convert_size<uint8_t, uint64_t>(inProbeDataSizeUncomprVecByte),
               witProbeComprState
            );
            uint8_t * outLAppendUncompr;
            uint8_t * outRAppendUncompr;
            std::tie(
               outSizeLComprByte, outLAppendUncompr, outLPtr
            ) = witProbeComprState.m_write_it_out_l_pos.done();
            std::tie(
               outSizeRComprByte, outRAppendUncompr, outRPtr
            ) = witProbeComprState.m_write_it_out_r_pos.done();
            outCountLog = witProbeComprState.m_write_it_out_l_pos.get_count_values();

            const size_t inProbeSizeScalarRemainderByte = inProbeSizeRestByte % vector_size_byte::value;
            if(inProbeSizeScalarRemainderByte) {
               typename natural_equi_join_probe_processing_unit_t<
                  scalar<v64<uint64_t>>,
                  t_data_structure,
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
                  natural_equi_join_probe_processing_unit_t,
                  t_data_structure,
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
               ) = witProbeUncomprState.m_write_it_out_l_pos.done();
               std::tie(
                  std::ignore, std::ignore, outRPtr
               ) = witProbeUncomprState.m_write_it_out_r_pos.done();
               outCountLog += witProbeUncomprState.m_write_it_out_l_pos.get_count_values();
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





}
#endif //MORPHSTORE_CORE_OPERATORS_OTFLY_DERECOMPR_JOIN_NATURAL_EQUI_COMPR_H
