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
 * @file calc_avx2.h
 * @brief This file contains calculation primitives for AVX2
 * @todo benchmarks: VL_BENCHMARK_CALC_AVX2_HADD_VARIANT_USING_SSE
 */


#ifndef MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_CALC_AVX2_H
#define MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_CALC_AVX2_H


#include <core/utils/logger.h>
#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/primitives/calc.h>

#include <functional>

namespace vectorlib{
   template<>
   struct add<avx2<v256<uint64_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::vector_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 64 bit integer values from two registers (avx2)" );
         return _mm256_add_epi64( p_vec1, p_vec2);
      }
   };
   template<>
   struct add<avx2<v256<uint32_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint32_t>>::vector_t
      apply(
         typename avx2<v256<uint32_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 32 bit integer values from two registers (avx2)" );
         return _mm256_add_epi32( p_vec1, p_vec2);
      }
   };
   template<>
   struct add<avx2<v256<uint16_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint16_t>>::vector_t
      apply(
         typename avx2<v256<uint16_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 16 bit integer values from two registers (avx2)" );
         return _mm256_add_epi16( p_vec1, p_vec2);
      }
   };
   template<>
   struct add<avx2<v256<uint8_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint8_t>>::vector_t
      apply(
         typename avx2<v256<uint8_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 8 bit integer values from two registers (avx2)" );
         return _mm256_add_epi8( p_vec1, p_vec2);
      }
   };

   //SIMDI SUBTRACTION
   template<>
   struct sub<avx2<v256<uint64_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::vector_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 64 bit integer values from two registers (avx2)" );
         return _mm256_sub_epi64( p_vec1, p_vec2);
      }
   };
   template<>
   struct sub<avx2<v256<uint32_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint32_t>>::vector_t
      apply(
         typename avx2<v256<uint32_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 32 bit integer values from two registers (avx2)" );
         return _mm256_sub_epi32( p_vec1, p_vec2);
      }
   };
   template<>
   struct sub<avx2<v256<uint16_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint16_t>>::vector_t
      apply(
         typename avx2<v256<uint16_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 16 bit integer values from two registers (avx2)" );
         return _mm256_sub_epi16( p_vec1, p_vec2);
      }
   };
   template<>
   struct sub<avx2<v256<uint8_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint8_t>>::vector_t
      apply(
         typename avx2<v256<uint8_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 8 bit integer values from two registers (avx2)" );
         return _mm256_sub_epi8( p_vec1, p_vec2);
      }
   };

   //SIMDI HORIZONTAL ADDITION
   template<>
   struct hadd<avx2<v256<uint64_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::base_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Horizontally add 64 bit integer values within one register (avx2)" );
         __m256i tmp =
            _mm256_castpd_si256(
               _mm256_hadd_pd(
                  _mm256_castsi256_pd(p_vec1),
                  _mm256_castsi256_pd(p_vec1)
               )
            );
         return _mm256_extract_epi64(tmp,0)+_mm256_extract_epi64(tmp,2);
      }
   };
   template<>
   struct hadd<avx2<v256<uint32_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint32_t>>::base_t
      apply(
         typename avx2<v256<uint32_t>>::vector_t const & p_vec1
      ) {
#ifdef VL_BENCHMARK_CALC_AVX2_HADD_VARIANT_USING_SSE
         trace( "[VECTOR] - Horizontally add 32 bit integer values within one register (avx2) using extract and sse intrinsics" );
         typename sse<v128<uint32_t>>::vector_t const tmp0 =
            _mm_add_epi32(
               _mm256_extracti128_si256( p_vec1, 0 ),
               _mm256_extracti128_si256( p_vec1, 1 )
            );
         /*__m128i tmp1 =
            _mm_hadd_epi32( tmp0, tmp0 );*/
         typename sse<v128<uint32_t>>::vector_t const tmp1 =
            _mm_add_epi32( tmp0, _mm_shuffle_epi32( tmp0, 0x4E ) );
         return _mm_extract_epi32( tmp1, 0 ) + _mm_extract_epi32( tmp1, 1 );
#else
         trace( "[VECTOR] - Horizontally add 32 bit integer values within one register (avx2)" );
         typename avx2<v256<uint32_t>>::vector_t const tmp0 =
            _mm256_hadd_epi32( p_vec1, p_vec1 );
         typename avx2<v256<uint32_t>>::vector_t const tmp1 =
            _mm256_hadd_epi32( tmp0, tmp0 );
         return _mm256_extract_epi32( tmp1, 0 ) + _mm256_extract_epi32( tmp1, 4 );
#endif
      }
   };
   template<>
   struct hadd<avx2<v256<uint16_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint16_t>>::base_t
      apply(
         typename avx2<v256<uint16_t>>::vector_t const & p_vec1
      ) {
#ifdef VL_BENCHMARK_CALC_AVX2_HADD_VARIANT_USING_SSE
         trace( "[VECTOR] - Horizontally add 16 bit integer values within one register (avx2) using extract and sse intrinsics" );
         typename sse<v128<uint16_t>>::vector_t const tmp0 =
            _mm_add_epi32(
               _mm256_extracti128_si256( p_vec1, 0 ),
               _mm256_extracti128_si256( p_vec1, 1 )
            );
         typename sse<v128<uint16_t>>::vector_t const tmp1 =
            _mm_add_epi16(
               tmp0,
               _mm_shuffle_epi32( tmp0, 0x4E )
            );
         typename sse<v128<uint16_t>>::vector_t const tmp2 =
            _mm_add_epi16(
               tmp1,
               _mm_shuffle_epi32( tmp1, 0xE1 )
            );
         return _mm_extract_epi16( tmp2, 0 ) + _mm_extract_epi16( tmp2, 1 );
#else
         trace( "[VECTOR] - Horizontally add 16 bit integer values within one register (avx2)" );
         typename avx2<v256<uint16_t>>::vector_t const tmp0 =
            _mm256_hadd_epi16( p_vec1, p_vec1 );
         typename avx2<v256<uint16_t>>::vector_t const tmp1 =
            _mm256_hadd_epi16( tmp0, tmp0 );
         typename avx2<v256<uint16_t>>::vector_t const tmp2 =
            _mm256_hadd_epi16( tmp1, tmp1 );
         return _mm256_extract_epi16( tmp2, 0 ) + _mm256_extract_epi16( tmp2, 8 );
#endif
      }
   };


   template<>
   struct hadd<avx2<v256<uint8_t>>> {
      //@todo: look out for better solution.
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint8_t>>::base_t
      apply(
         typename avx2<v256<uint8_t>>::vector_t const & p_vec1
      ) {
         typename sse<v128<uint8_t>>::vector_t const tmp0 =
            _mm_add_epi8(
               _mm256_extracti128_si256( p_vec1, 0 ),
               _mm256_extracti128_si256( p_vec1, 1 )
            );
         typename sse<v128<uint8_t>>::vector_t const tmp1 =
            _mm_add_epi8(
               tmp0,
               _mm_shuffle_epi32( tmp0, 0x4E)
            );
         typename sse<v128<uint8_t>>::vector_t const tmp2 =
            _mm_add_epi8(
               tmp1,
               _mm_shuffle_epi32( tmp1, 0x39 )
            );
         typename sse<v128<uint8_t>>::vector_t  const tmp3 =
            _mm_add_epi8(
               tmp2,
               _mm_shufflelo_epi16( tmp2, 0x1 )
            );
         return _mm_extract_epi8( tmp3, 0 ) + _mm_extract_epi8( tmp3, 1 );
      }
   };

   //SIMDI MULTIPLICATION
   template<>
   struct mul<avx2<v256<uint64_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::vector_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (avx2)" );
         info( "[VECTOR] - _mm256_mul_epu32 is called (only the lower 32 bit are actually processed" );
         return _mm256_mul_epu32( p_vec1, p_vec2);
      }
   };


   template<>
   struct div<avx2<v256<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::vector_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Divide 64 bit integer values from two registers (avx2)" );
         __m256d divhelper = _mm256_set1_pd(0x0010000000000000);

         return
            _mm256_xor_si256(
               _mm256_castpd_si256(
                  _mm256_add_pd(
                     _mm256_div_pd(
                        _mm256_castsi256_pd(p_vec1),
                        _mm256_castsi256_pd(p_vec2)
                     ),
                     divhelper
                  )
               ),
               _mm256_castpd_si256(
                  divhelper
               )
            );
      }
   };
   template<>
   struct mod<avx2<v256<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::vector_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Modulo divide 64 bit integer values from two registers (avx2)" );
         info( "[VECTOR] - MODULO IS A WORKAROUND" );
         __m256d divhelper = _mm256_set1_pd(0x0010000000000000);
         __m256d intermediate =
            _mm256_add_pd(
               _mm256_floor_pd(
                  _mm256_div_pd(
                     _mm256_castsi256_pd(p_vec1),
                     _mm256_castsi256_pd(p_vec2)
                  )
               ),
               divhelper
            );
         return
            _mm256_sub_epi64(
               p_vec1,
               _mm256_mul_epi32(
                  _mm256_xor_si256(
                     _mm256_castpd_si256(intermediate),
                     _mm256_castpd_si256(divhelper)
                  ),
                  p_vec2
               )
            );
      }
   };
   template<>
   struct inv<avx2<v256<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::vector_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 64 bit integer values of one register (avx2)" );
         return _mm256_sub_epi64( _mm256_set1_epi64x(0), p_vec1);
      }
   };
   template<>
   struct shift_left<avx2<v256<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::vector_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (all by the same distance) (avx2)" );
         return _mm256_slli_epi64(p_vec1, p_distance);
      }
   };
   template<>
   struct shift_left_individual<avx2<v256<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::vector_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_data,
         typename avx2<v256<uint64_t>>::vector_t const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (each by its individual distance) (avx2)" );
         return _mm256_sllv_epi64(p_data, p_distance);
      }
   };
   template<>
   struct shift_right<avx2<v256<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::vector_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (all by the same distance) (avx2)" );
         return _mm256_srli_epi64(p_vec1, p_distance);
      }
   };
   template<>
   struct shift_right_individual<avx2<v256<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::vector_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_data,
         typename avx2<v256<uint64_t>>::vector_t const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (each by its individual distance) (avx2)" );
         return _mm256_srlv_epi64(p_data, p_distance);
      }
   };
}

#endif /* MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_CALC_AVX2_H */

