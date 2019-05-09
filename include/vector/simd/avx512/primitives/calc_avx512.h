/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   calc_avx512.h
 * Author: Annett
 *
 * Created on 17. April 2019, 11:07
 */

#ifndef CALC_AVX512_H
#define CALC_AVX512_H



#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx512/extension_avx512.h>
#include <vector/primitives/calc.h>

#include <functional>

namespace vector{
   template<>
   struct add<avx512<v512<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_>>::vector_t
      apply(
         typename avx512<v512<uint64_>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 64 bit integer values from two registers (avx512)" );
         return _mm512_add_epi64( p_vec1, p_vec2);
      }
   };
   template<>
   struct sub<avx512<v512<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_>>::vector_t
      apply(
         typename avx512<v512<uint64_>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 64 bit integer values from two registers (avx512)" );
         return _mm512_sub_epi64( p_vec1, p_vec2);
      }
   };
   template<>
   struct hadd<avx512<v512<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_>>::base_t
      apply(
         typename avx512<v512<uint64_>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Horizontally add 64 bit integer values one register (avx512)" );
         return _mm512_reduce_add_epi64(p_vec1);
      }
   };
   template<>
   struct mul<avx512<v512<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_>>::vector_t
      apply(
         typename avx512<v512<uint64_>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (avx512)" );
         warn( "[VECTOR] - _mm512_mul_epu32 is called (only the lower 32 bit are actually processed" );
         return _mm512_mul_epu32( p_vec1, p_vec2);
      }
   };
   template<>
   struct div<avx512<v512<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_>>::vector_t
      apply(
         typename avx512<v512<uint64_>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Divide 64 bit integer values from two registers (avx512)" );
         __m512d divhelper = _mm512_set1_pd(0x0010000000000000);
         return
            _mm512_xor_si512(
               _mm512_castpd_si512(
                  _mm512_add_pd(
                     _mm512_div_pd(
                        _mm512_castsi512_pd(p_vec1),
                        _mm512_castsi512_pd(p_vec2)
                     ),
                     divhelper
                  )
               ),
               _mm512_castpd_si512(
                  divhelper
               )
            );
      }
   };
   template<>
   struct mod<avx512<v512<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_>>::vector_t
      apply(
         typename avx512<v512<uint64_>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Modulo divide 64 bit integer values from two registers (avx512)" );
         warn( "[VECTOR] - MODULO IS A WORKAROUND" );
         __m512d divhelper = _mm512_set1_pd(0x0010000000000000);
         __m512d intermediate =
            _mm512_add_pd(
               _mm512_floor_pd(
                  _mm512_div_pd(
                     _mm512_castsi512_pd(p_vec1),
                     _mm512_castsi512_pd(p_vec2)
                  )
               ),
               divhelper
            );
         return
            _mm512_sub_epi64(
               p_vec1,
               _mm512_mul_epi32(
                  _mm512_xor_si512(
                     _mm512_castpd_si512(intermediate),
                     _mm512_castpd_si512(divhelper)
                  ),
                  p_vec2
               )
            );
      }
   };
   template<>
   struct inv<avx512<v512<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_>>::vector_t
      apply(
         typename avx512<v512<uint64_>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 64 bit integer values of one register (avx512)" );
         return _mm512_sub_epi64( _mm512_set1_epi64(0), p_vec1);
      }
   };
}
#endif /* CALC_AVX512_H */

