/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   calc_sse.h
 * Author: Annett
 *
 * Created on 17. April 2019, 11:07
 */

#ifndef CALC_SSE_H
#define CALC_SSE_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/sse/extension_sse.h>
#include <vector/primitives/calc.h>

#include <functional>

namespace vector{

   template<>
   struct add<sse<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint64_t>>::vector_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1,
         typename sse<v128<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 64 bit integer values from two registers (sse)" );
         return _mm_add_epi64( p_vec1, p_vec2);
      }
   };
   template<>
   struct sub<sse<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint64_t>>::vector_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1,
         typename sse<v128<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 64 bit integer values from two registers (sse)" );
         return _mm_sub_epi64( p_vec1, p_vec2);
      }
   };
   template<>
   struct hadd<sse<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint64_t>>::base_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Horizontally add 64 bit integer values one register (sse)" );
         return
            _mm_extract_epi64(
               _mm_castpd_si128(
                  _mm_hadd_pd(
                     _mm_castsi128_pd(p_vec1),
                     _mm_castsi128_pd(p_vec1)
                  )
               ),
               0
            );
      }
   };
   template<>
   struct mul<sse<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint64_t>>::vector_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1,
         typename sse<v128<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (sse)" );
         warn( "[VECTOR] - _mm_mul_epu32 is called (only the lower 32 bit are actually processed" );
         return _mm_mul_epu32( p_vec1, p_vec2);
      }
   };

   template<>
   struct div<sse<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint64_t>>::vector_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const &p_vec1,
         typename sse<v128<uint64_t>>::vector_t const &p_vec2
      ) {
         trace("[VECTOR] - Divide 64 bit integer values from two registers (sse)");
         __m128d intermediate;
         __m128d divhelper=_mm_set1_pd(0x0010000000000000);

         return
            _mm_xor_si128(
               _mm_castpd_si128(
                  _mm_add_pd(
                     _mm_div_pd(
                        _mm_castsi128_pd(p_vec1),
                        _mm_castsi128_pd(p_vec2)
                     ),
                     divhelper
                  )
               ),
               _mm_castpd_si128(
                  divhelper
               )
            );
      }
   };

   template<>
   struct mod<sse<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint64_t>>::vector_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1,
         typename sse<v128<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Modulo divide 64 bit integer values from two registers (sse)" );
         warn( "[VECTOR] - MODULO IS A WORKAROUND" );
         __m128d divhelper = _mm_set1_pd(0x0010000000000000);
         __m128d intermediate =
            _mm_add_pd(
               _mm_floor_pd(
                  _mm_div_pd(
                     _mm_castsi128_pd(p_vec1),
                     _mm_castsi128_pd(p_vec2)
                  )
               ),
               divhelper
            );
         return
            _mm_sub_epi64(
               p_vec1,
               _mm_mul_epi32(
                  _mm_xor_si128(
                     _mm_castpd_si128(intermediate),
                     _mm_castpd_si128(divhelper)
                  ),
                  p_vec2
               )
            );
      }
   };

   template<>
   struct inv<sse<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint64_t>>::vector_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 64 bit integer values of one register (sse)" );
         return _mm_sub_epi64( _mm_set1_epi64x(0), p_vec1);
      }
   };

}
#endif /* CALC_SSE_H */

