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

#ifndef CALC_AVX2_H
#define CALC_AVX2_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/primitives/calc.h>

#include <functional>

namespace vector{
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
   struct mul<avx2<v256<uint64_t>>/*, 64*/> {
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
}

#endif /* CALC_AVX2_H */

