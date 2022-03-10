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

#ifndef MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_CALC_SSE_H
#define MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_CALC_SSE_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include "../extension_sse.h"
#include "../../../primitives/calc.h"

#include <functional>

namespace vectorlib{

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
   struct min<sse<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint64_t>>::vector_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1,
         typename sse<v128<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - build minimum of 64 bit integer values from two registers (sse)" );
         return _mm_blendv_epi8(p_vec2, p_vec1, _mm_cmpgt_epi64(p_vec2, p_vec1));
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
//         info( "[VECTOR] - _mm_mul_epu32 is called (only the lower 32 bit are actually processed" );
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
         __m128d divhelper=_mm_set1_pd(0x0010000000000000);

         return
            _mm_xor_si128(
               _mm_castpd_si128(
                  _mm_add_pd(
                     _mm_floor_pd(
                        _mm_div_pd(
                           _mm_castsi128_pd(p_vec1),
                           _mm_castsi128_pd(p_vec2)
                        )
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
         info( "[VECTOR] - MODULO IS A WORKAROUND" );
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

   template<>
   struct shift_left<sse<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint64_t>>::vector_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (all by the same distance) (sse)" );
         return _mm_slli_epi64(p_vec1, p_distance);
      }
   };

   template<>
   struct shift_left_individual<sse<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint64_t>>::vector_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_data,
         typename sse<v128<uint64_t>>::vector_t const & p_distance
      ){
         // SSE does not have an intrinsic for this.
         // The comparison with 64 is necessary, since the scalar shift behaves
         // strangely in that case.
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (each by its individual distance) (sse)" );
         uint64_t distance0 = _mm_extract_epi64(p_distance, 0);
         uint64_t distance1 = _mm_extract_epi64(p_distance, 1);
         return _mm_set_epi64x(
                 (distance1 == 64) ? 0 : (_mm_extract_epi64(p_data, 1) << distance1),
                 (distance0 == 64) ? 0 : (_mm_extract_epi64(p_data, 0) << distance0)
         );
      }
   };

   template<>
   struct shift_right<sse<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint64_t>>::vector_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (all by the same distance) (sse)" );
         return _mm_srli_epi64(p_vec1, p_distance);
      }
   };

   template<>
   struct shift_right_individual<sse<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint64_t>>::vector_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_data,
         typename sse<v128<uint64_t>>::vector_t const & p_distance
      ){
         // SSE does not have an intrinsic for this.
         // The comparison with 64 is necessary, since the scalar shift behaves
         // strangely in that case.
         // The static_cast to an unsigned type is necessary, since the scalar
         // shift shifts in sign-bits otherwise.
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (each by its individual distance) (sse)" );
         uint64_t distance0 = _mm_extract_epi64(p_distance, 0);
         uint64_t distance1 = _mm_extract_epi64(p_distance, 1);
         return _mm_set_epi64x(
                 (distance1 == 64) ? 0 : (static_cast<uint64_t>(_mm_extract_epi64(p_data, 1)) >> distance1),
                 (distance0 == 64) ? 0 : (static_cast<uint64_t>(_mm_extract_epi64(p_data, 0)) >> distance0)
         );
      }
   };

   /*NOTE: This primitive automatically substracts the unused bits, where a bitmask is larger than required*/
   template<typename T>
   struct count_leading_zero<sse<v128<T>>> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static uint8_t
      apply(
         typename sse<v128<U>>::mask_t const & p_mask
      ) {
  
       //  return __builtin_clz(p_mask)-(sizeof(p_mask)*8-sse<v128<U>>::vector_helper_t::element_count::value);
           return __builtin_clz(p_mask)-(sizeof(p_mask)*8-sse<v128<U>>::vector_helper_t::element_count::value) - (sizeof(p_mask) > 4 ? 0 : 16);
          
      }
   };
}
#endif /* MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_CALC_SSE_H */

