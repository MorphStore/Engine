/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   calc_avx2.h
 * Author: Annett
 *
 * Created on 17. April 2019, 11:07
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
   struct add<avx2<v256<uint64_t>>/*, 64*/> {
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
   struct min<avx2<v256<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::vector_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - build minimum of 64 bit integer values from two registers (sse)" );
         return _mm256_blendv_epi8(p_vec2, p_vec1, _mm256_cmpgt_epi64(p_vec2, p_vec1));
      }
   };
   
   template<>
   struct sub<avx2<v256<uint64_t>>/*, 64*/> {
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
   struct hadd<avx2<v256<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::base_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Horizontally add 64 bit integer values one register (avx2)" );
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
   struct hor<avx2<v256<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::base_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Horizontally or 64 bit integer values one register (avx2)" );
         return  _mm256_extract_epi64(p_vec1, 0) |
                 _mm256_extract_epi64(p_vec1, 1) |
                 _mm256_extract_epi64(p_vec1, 2) |
                 _mm256_extract_epi64(p_vec1, 3);
      }
   };
   template<>
   struct mul<avx2<v256<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint64_t>>::vector_t
      apply(
         typename avx2<v256<uint64_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (avx2)" );
          /// @todo Fix info call
         //info( "[VECTOR] - _mm256_mul_epu32 is called (only the lower 32 bit are actually processed" );
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
                     _mm256_floor_pd(
                        _mm256_div_pd(
                           _mm256_castsi256_pd(p_vec1),
                           _mm256_castsi256_pd(p_vec2)
                        )
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
         //info( "[VECTOR] - MODULO IS A WORKAROUND" );
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
   
      /*NOTE: This primitive automatically substracts the unused bits, where a bitmask is larger than required*/
   template<typename T>
   struct count_leading_zero<avx2<v256<T>>> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static uint8_t
      apply(
         typename avx2<v256<U>>::mask_t const & p_mask
      ) {

         //return __builtin_clz(p_mask)-(sizeof(p_mask)*8-avx2<v256<U>>::vector_helper_t::element_count::value);
         return __builtin_clz(p_mask)-(sizeof(p_mask)*8-avx2<v256<U>>::vector_helper_t::element_count::value) - (sizeof(p_mask) > 4 ? 0 : 16);
      }
   };
   
}

#endif /* MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_CALC_AVX2_H */

