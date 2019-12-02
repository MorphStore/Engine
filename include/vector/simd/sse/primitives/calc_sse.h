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
#include <vector/simd/sse/extension_sse.h>
#include <vector/primitives/calc.h>

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
         info( "[VECTOR] - _mm_mul_epu32 is called (only the lower 32 bit are actually processed" );
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


   template<>
   struct add<sse<v128<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint32_t>>::vector_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_vec1,
         typename sse<v128<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 32 bit integer values from two registers (sse)" );
         return _mm_add_epi32( p_vec1, p_vec2);
      }
   };

   template<>
   struct sub<sse<v128<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint32_t>>::vector_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_vec1,
         typename sse<v128<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 32 bit integer values from two registers (sse)" );
         return _mm_sub_epi32( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct min<sse<v128<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint32_t>>::vector_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_vec1,
         typename sse<v128<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - build minimum of 32 bit integer values from two registers (sse)" );
         return _mm_blendv_epi8(p_vec2, p_vec1, _mm_cmpgt_epi32(p_vec2, p_vec1));
      }
   };
      
   //doesn't work yet, error: cannot convert ‘vectorlib::sse<vectorlib::vector_view<128, unsigned int> >::base_t’ {aka ‘unsigned int’} 
   //to ‘vector_t’ {aka ‘__vector(2) long long int’} in initialization   
   // template<>
   // struct hadd<sse<v128<uint32_t>>/*, 32*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename sse<v128<uint32_t>>::base_t
   //    apply(
   //       typename sse<v128<uint32_t>>::vector_t const & p_vec1
   //    ){
   //       trace( "[VECTOR] - Horizontally add 32 bit integer values one register (sse)" );
   //       return
   //          _mm_extract_epi32(
   //             _mm_castpd_si128(
   //                _mm_hadd_pd(
   //                   _mm_castsi128_pd(p_vec1),
   //                   _mm_castsi128_pd(p_vec1)
   //                )
   //             ),
   //             0
   //          );
   //    }
   // };

   template<>
   struct mul<sse<v128<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint32_t>>::vector_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_vec1,
         typename sse<v128<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 32 bit integer values from two registers (sse)" );
         info( "[VECTOR] - _mm_mullo_epi32 is called" );
         return _mm_mullo_epi32( p_vec1, p_vec2);
      }
   };
   //doesn't work yet
   // template<>
   // struct div<sse<v128<uint32_t>>/*, 32*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename sse<v128<uint32_t>>::vector_t
   //    apply(
   //       typename sse<v128<uint32_t>>::vector_t const &p_vec1,
   //       typename sse<v128<uint32_t>>::vector_t const &p_vec2
   //    ) {
   //       trace("[VECTOR] - Divide 32 bit integer values from two registers (sse)");
   //       __m128d divhelper=_mm_set1_pd(0x0010000000000000);

   //       return
   //          _mm_xor_si128(
   //             _mm_castpd_si128(
   //                _mm_add_pd(
   //                   _mm_floor_pd(
   //                      _mm_div_pd(
   //                         _mm_castsi128_pd(p_vec1),
   //                         _mm_castsi128_pd(p_vec2)
   //                      )
   //                   ),
   //                   divhelper
   //                )
   //             ),
   //             _mm_castpd_si128(
   //                divhelper
   //             )
   //          );
   //    }
   // };
   //doesn't work yet
   // template<>
   // struct mod<sse<v128<uint32_t>>/*, 32*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename sse<v128<uint32_t>>::vector_t
   //    apply(
   //       typename sse<v128<uint32_t>>::vector_t const & p_vec1,
   //       typename sse<v128<uint32_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Modulo divide 32 bit integer values from two registers (sse)" );
   //       info( "[VECTOR] - MODULO IS A WORKAROUND" );
   //       __m128d divhelper = _mm_set1_pd(0x0010000000000000);
   //       __m128d intermediate =
   //          _mm_add_pd(
   //             _mm_floor_pd(
   //                _mm_div_pd(
   //                   _mm_castsi128_pd(p_vec1),
   //                   _mm_castsi128_pd(p_vec2)
   //                )
   //             ),
   //             divhelper
   //          );
   //       return
   //          _mm_sub_epi32(
   //             p_vec1,
   //             _mm_mul_epi32(
   //                _mm_xor_si128(
   //                   _mm_castpd_si128(intermediate),
   //                   _mm_castpd_si128(divhelper)
   //                ),
   //                p_vec2
   //             )
   //          );
   //    }
   // };
   //not tested
   template<>
   struct inv<sse<v128<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint32_t>>::vector_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 32 bit integer values of one register (sse)" );
         return _mm_sub_epi32( _mm_set1_epi32(0), p_vec1);
      }
   };
   //not tested
   template<>
   struct shift_left<sse<v128<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint32_t>>::vector_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 32 bit integer values of one register (all by the same distance) (sse)" );
         return _mm_slli_epi32(p_vec1, p_distance);
      }
   };
   //not tested
   template<>
   struct shift_left_individual<sse<v128<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint32_t>>::vector_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_data,
         typename sse<v128<uint32_t>>::vector_t const & p_distance
      ){
         // SSE does not have an intrinsic for this.
         //is the comparison with 32 necessary?
         trace( "[VECTOR] - Left-shifting 32 bit integer values of one register (each by its individual distance) (sse)" );
         uint32_t distance0 = _mm_extract_epi32(p_distance, 0);
         uint32_t distance1 = _mm_extract_epi32(p_distance, 1);
         uint32_t distance2 = _mm_extract_epi32(p_distance, 2);
         uint32_t distance3 = _mm_extract_epi32(p_distance, 3);
         return _mm_set_epi32(
                 (distance3 == 32) ? 0 : (_mm_extract_epi32(p_data, 3) << distance3),
                 (distance2 == 32) ? 0 : (_mm_extract_epi32(p_data, 2) << distance2),
                 (distance1 == 32) ? 0 : (_mm_extract_epi32(p_data, 1) << distance1),
                 (distance0 == 32) ? 0 : (_mm_extract_epi32(p_data, 0) << distance0)

         );
      }
   };
   //not tested
   template<>
   struct shift_right<sse<v128<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint32_t>>::vector_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 32 bit integer values of one register (all by the same distance) (sse)" );
         return _mm_srli_epi32(p_vec1, p_distance);
      }
   };
   //not tested
   template<>
   struct shift_right_individual<sse<v128<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint32_t>>::vector_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_data,
         typename sse<v128<uint32_t>>::vector_t const & p_distance
      ){
         // SSE does not have an intrinsic for this.
         // is the comparison with 32 necessary?
         // The static_cast to an unsigned type is necessary, since the scalar
         // shift shifts in sign-bits otherwise.
         trace( "[VECTOR] - Right-shifting 32 bit integer values of one register (each by its individual distance) (sse)" );
         uint32_t distance0 = _mm_extract_epi32(p_distance, 0);
         uint32_t distance1 = _mm_extract_epi32(p_distance, 1);
         uint32_t distance2 = _mm_extract_epi32(p_distance, 2);
         uint32_t distance3 = _mm_extract_epi32(p_distance, 3);
         return _mm_set_epi32(
                 (distance3 == 32) ? 0 : (static_cast<uint32_t>(_mm_extract_epi32(p_data, 3)) >> distance3),
                 (distance2 == 32) ? 0 : (static_cast<uint32_t>(_mm_extract_epi32(p_data, 2)) >> distance2),
                 (distance1 == 32) ? 0 : (static_cast<uint32_t>(_mm_extract_epi32(p_data, 1)) >> distance1),
                 (distance0 == 32) ? 0 : (static_cast<uint32_t>(_mm_extract_epi32(p_data, 0)) >> distance0)
         );
      }
   };

   template<>
   struct add<sse<v128<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint16_t>>::vector_t
      apply(
         typename sse<v128<uint16_t>>::vector_t const & p_vec1,
         typename sse<v128<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 16 bit integer values from two registers (sse)" );
         return _mm_add_epi16( p_vec1, p_vec2);
      }
   };

   template<>
   struct sub<sse<v128<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint16_t>>::vector_t
      apply(
         typename sse<v128<uint16_t>>::vector_t const & p_vec1,
         typename sse<v128<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 16 bit integer values from two registers (sse)" );
         return _mm_sub_epi16( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct min<sse<v128<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint16_t>>::vector_t
      apply(
         typename sse<v128<uint16_t>>::vector_t const & p_vec1,
         typename sse<v128<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - build minimum of 16 bit integer values from two registers (sse)" );
         return _mm_blendv_epi8(p_vec2, p_vec1, _mm_cmpgt_epi16(p_vec2, p_vec1));
      }
   };
      
   //doesn't work yet
   // template<>
   // struct hadd<sse<v128<uint16_t>>/*, 16*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename sse<v128<uint16_t>>::base_t
   //    apply(
   //       typename sse<v128<uint16_t>>::vector_t const & p_vec1
   //    ){
   //       trace( "[VECTOR] - Horizontally add 16 bit integer values one register (sse)" );
   //       return
   //          _mm_extract_epi16(
   //             _mm_castpd_si128(
   //                _mm_hadd_pd(
   //                   _mm_castsi128_pd(p_vec1),
   //                   _mm_castsi128_pd(p_vec1)
   //                )
   //             ),
   //             0
   //          );
   //    }
   // };

   template<>
   struct mul<sse<v128<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint16_t>>::vector_t
      apply(
         typename sse<v128<uint16_t>>::vector_t const & p_vec1,
         typename sse<v128<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 16 bit integer values from two registers (sse)" );
         info( "[VECTOR] - _mm_mullo_epi16 is called" );
         return _mm_mullo_epi16( p_vec1, p_vec2);
      }
   };
   //doesn't work yet
   // template<>
   // struct div<sse<v128<uint16_t>>/*, 16*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename sse<v128<uint16_t>>::vector_t
   //    apply(
   //       typename sse<v128<uint16_t>>::vector_t const &p_vec1,
   //       typename sse<v128<uint16_t>>::vector_t const &p_vec2
   //    ) {
   //       trace("[VECTOR] - Divide 16 bit integer values from two registers (sse)");
   //       __m128d divhelper=_mm_set1_pd(0x0010000000000000);

   //       return
   //          _mm_xor_si128(
   //             _mm_castpd_si128(
   //                _mm_add_pd(
   //                   _mm_floor_pd(
   //                      _mm_div_pd(
   //                         _mm_castsi128_pd(p_vec1),
   //                         _mm_castsi128_pd(p_vec2)
   //                      )
   //                   ),
   //                   divhelper
   //                )
   //             ),
   //             _mm_castpd_si128(
   //                divhelper
   //             )
   //          );
   //    }
   // };
   //doesn't work yet
   // template<>
   // struct mod<sse<v128<uint16_t>>/*, 16*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename sse<v128<uint16_t>>::vector_t
   //    apply(
   //       typename sse<v128<uint16_t>>::vector_t const & p_vec1,
   //       typename sse<v128<uint16_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Modulo divide 16 bit integer values from two registers (sse)" );
   //       info( "[VECTOR] - MODULO IS A WORKAROUND" );
   //       __m128d divhelper = _mm_set1_pd(0x0010000000000000);
   //       __m128d intermediate =
   //          _mm_add_pd(
   //             _mm_floor_pd(
   //                _mm_div_pd(
   //                   _mm_castsi128_pd(p_vec1),
   //                   _mm_castsi128_pd(p_vec2)
   //                )
   //             ),
   //             divhelper
   //          );
   //       return
   //          _mm_sub_epi32(
   //             p_vec1,
   //             _mm_mul_epi32(
   //                _mm_xor_si128(
   //                   _mm_castpd_si128(intermediate),
   //                   _mm_castpd_si128(divhelper)
   //                ),
   //                p_vec2
   //             )
   //          );
   //    }
   // };
   //not tested
   template<>
   struct inv<sse<v128<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint16_t>>::vector_t
      apply(
         typename sse<v128<uint16_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 16 bit integer values of one register (sse)" );
         return _mm_sub_epi16( _mm_set1_epi16(0), p_vec1);
      }
   };
   //not tested
   template<>
   struct shift_left<sse<v128<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint16_t>>::vector_t
      apply(
         typename sse<v128<uint16_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 16 bit integer values of one register (all by the same distance) (sse)" );
         return _mm_slli_epi16(p_vec1, p_distance);
      }
   };
   //not tested
   template<>
   struct shift_left_individual<sse<v128<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint16_t>>::vector_t
      apply(
         typename sse<v128<uint16_t>>::vector_t const & p_data,
         typename sse<v128<uint16_t>>::vector_t const & p_distance
      ){
         // SSE does not have an intrinsic for this.
         //is the comparison with 16 necessary?
         trace( "[VECTOR] - Left-shifting 16 bit integer values of one register (each by its individual distance) (sse)" );
         uint16_t distance0 = _mm_extract_epi16(p_distance, 0);
         uint16_t distance1 = _mm_extract_epi16(p_distance, 1);
         uint16_t distance2 = _mm_extract_epi16(p_distance, 2);
         uint16_t distance3 = _mm_extract_epi16(p_distance, 3);
         uint16_t distance4 = _mm_extract_epi16(p_distance, 4);
         uint16_t distance5 = _mm_extract_epi16(p_distance, 5);
         uint16_t distance6 = _mm_extract_epi16(p_distance, 6);
         uint16_t distance7 = _mm_extract_epi16(p_distance, 7);
         return _mm_set_epi16(
                 (distance7 == 16) ? 0 : (_mm_extract_epi16(p_data, 7) << distance7),
                 (distance6 == 16) ? 0 : (_mm_extract_epi16(p_data, 6) << distance6),
                 (distance5 == 16) ? 0 : (_mm_extract_epi16(p_data, 5) << distance5),
                 (distance4 == 16) ? 0 : (_mm_extract_epi16(p_data, 4) << distance4),
                 (distance3 == 16) ? 0 : (_mm_extract_epi16(p_data, 3) << distance3),
                 (distance2 == 16) ? 0 : (_mm_extract_epi16(p_data, 2) << distance2),
                 (distance1 == 16) ? 0 : (_mm_extract_epi16(p_data, 1) << distance1),
                 (distance0 == 16) ? 0 : (_mm_extract_epi16(p_data, 0) << distance0)

         );
      }
   };
   //not tested
   template<>
   struct shift_right<sse<v128<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint16_t>>::vector_t
      apply(
         typename sse<v128<uint16_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 16 bit integer values of one register (all by the same distance) (sse)" );
         return _mm_srli_epi16(p_vec1, p_distance);
      }
   };
   // //not tested
   template<>
   struct shift_right_individual<sse<v128<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint16_t>>::vector_t
      apply(
         typename sse<v128<uint16_t>>::vector_t const & p_data,
         typename sse<v128<uint16_t>>::vector_t const & p_distance
      ){
         // SSE does not have an intrinsic for this.
         // is the comparison with 16 necessary?
         // The static_cast to an unsigned type is necessary, since the scalar
         // shift shifts in sign-bits otherwise.
         trace( "[VECTOR] - Right-shifting 16 bit integer values of one register (each by its individual distance) (sse)" );
         uint16_t distance0 = _mm_extract_epi16(p_distance, 0);
         uint16_t distance1 = _mm_extract_epi16(p_distance, 1);
         uint16_t distance2 = _mm_extract_epi16(p_distance, 2);
         uint16_t distance3 = _mm_extract_epi16(p_distance, 3);
         uint16_t distance4 = _mm_extract_epi16(p_distance, 4);
         uint16_t distance5 = _mm_extract_epi16(p_distance, 5);
         uint16_t distance6 = _mm_extract_epi16(p_distance, 6);
         uint16_t distance7 = _mm_extract_epi16(p_distance, 7);
         return _mm_set_epi16(
                 (distance7 == 16) ? 0 : (static_cast<uint16_t>(_mm_extract_epi16(p_data, 3)) >> distance7),
                 (distance6 == 16) ? 0 : (static_cast<uint16_t>(_mm_extract_epi16(p_data, 2)) >> distance6),
                 (distance5 == 16) ? 0 : (static_cast<uint16_t>(_mm_extract_epi16(p_data, 1)) >> distance5),
                 (distance4 == 16) ? 0 : (static_cast<uint16_t>(_mm_extract_epi16(p_data, 0)) >> distance4),
                 (distance3 == 16) ? 0 : (static_cast<uint16_t>(_mm_extract_epi16(p_data, 3)) >> distance3),
                 (distance2 == 16) ? 0 : (static_cast<uint16_t>(_mm_extract_epi16(p_data, 2)) >> distance2),
                 (distance1 == 16) ? 0 : (static_cast<uint16_t>(_mm_extract_epi16(p_data, 1)) >> distance1),
                 (distance0 == 16) ? 0 : (static_cast<uint16_t>(_mm_extract_epi16(p_data, 0)) >> distance0)
         );
      }
   };


   template<>
   struct add<sse<v128<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint8_t>>::vector_t
      apply(
         typename sse<v128<uint8_t>>::vector_t const & p_vec1,
         typename sse<v128<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 16 bit integer values from two registers (sse)" );
         return _mm_add_epi8( p_vec1, p_vec2);
      }
   };

   template<>
   struct sub<sse<v128<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint8_t>>::vector_t
      apply(
         typename sse<v128<uint8_t>>::vector_t const & p_vec1,
         typename sse<v128<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 8 bit integer values from two registers (sse)" );
         return _mm_sub_epi8( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct min<sse<v128<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint8_t>>::vector_t
      apply(
         typename sse<v128<uint8_t>>::vector_t const & p_vec1,
         typename sse<v128<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - build minimum of 8 bit integer values from two registers (sse)" );
         return _mm_blendv_epi8(p_vec2, p_vec1, _mm_cmpgt_epi8(p_vec2, p_vec1));
      }
   };
      
   //doesn't work yet
   // template<>
   // struct hadd<sse<v128<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename sse<v128<uint8_t>>::base_t
   //    apply(
   //       typename sse<v128<uint8_t>>::vector_t const & p_vec1
   //    ){
   //       trace( "[VECTOR] - Horizontally add 8 bit integer values one register (sse)" );
   //       return
   //          _mm_extract_epi8(
   //             _mm_castpd_si128(
   //                _mm_hadd_pd(
   //                   _mm_castsi128_pd(p_vec1),
   //                   _mm_castsi128_pd(p_vec1)
   //                )
   //             ),
   //             0
   //          );
   //    }
   // };
   //doesn't work, no easy 8bit intrinsic to replace _mm_mullo_epi16
   // template<>
   // struct mul<sse<v128<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename sse<v128<uint8_t>>::vector_t
   //    apply(
   //       typename sse<v128<uint8_t>>::vector_t const & p_vec1,
   //       typename sse<v128<uint8_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Multiply 8 bit integer values from two registers (sse)" );
   //       info( "[VECTOR] - _mm_mullo_epi16 is called" );
   //       return _mm_mullo_epi16( p_vec1, p_vec2);
   //    }
   // };
   //doesn't work yet
   // template<>
   // struct div<sse<v128<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename sse<v128<uint8_t>>::vector_t
   //    apply(
   //       typename sse<v128<uint8_t>>::vector_t const &p_vec1,
   //       typename sse<v128<uint8_t>>::vector_t const &p_vec2
   //    ) {
   //       trace("[VECTOR] - Divide 8 bit integer values from two registers (sse)");
   //       __m128d divhelper=_mm_set1_pd(0x0010000000000000);

   //       return
   //          _mm_xor_si128(
   //             _mm_castpd_si128(
   //                _mm_add_pd(
   //                   _mm_floor_pd(
   //                      _mm_div_pd(
   //                         _mm_castsi128_pd(p_vec1),
   //                         _mm_castsi128_pd(p_vec2)
   //                      )
   //                   ),
   //                   divhelper
   //                )
   //             ),
   //             _mm_castpd_si128(
   //                divhelper
   //             )
   //          );
   //    }
   // };
   //doesn't work yet
   // template<>
   // struct mod<sse<v128<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename sse<v128<uint8_t>>::vector_t
   //    apply(
   //       typename sse<v128<uint8_t>>::vector_t const & p_vec1,
   //       typename sse<v128<uint8_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Modulo divide 8 bit integer values from two registers (sse)" );
   //       info( "[VECTOR] - MODULO IS A WORKAROUND" );
   //       __m128d divhelper = _mm_set1_pd(0x0010000000000000);
   //       __m128d intermediate =
   //          _mm_add_pd(
   //             _mm_floor_pd(
   //                _mm_div_pd(
   //                   _mm_castsi128_pd(p_vec1),
   //                   _mm_castsi128_pd(p_vec2)
   //                )
   //             ),
   //             divhelper
   //          );
   //       return
   //          _mm_sub_epi32(
   //             p_vec1,
   //             _mm_mul_epi32(
   //                _mm_xor_si128(
   //                   _mm_castpd_si128(intermediate),
   //                   _mm_castpd_si128(divhelper)
   //                ),
   //                p_vec2
   //             )
   //          );
   //    }
   // };
   //not tested
   template<>
   struct inv<sse<v128<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint8_t>>::vector_t
      apply(
         typename sse<v128<uint8_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 8 bit integer values of one register (sse)" );
         return _mm_sub_epi8( _mm_set1_epi8(0), p_vec1);
      }
   };
   //doesn't work, no easy 8bit intrinsic to replace _mm_slli_epi16
   // template<>
   // struct shift_left<sse<v128<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename sse<v128<uint8_t>>::vector_t
   //    apply(
   //       typename sse<v128<uint8_t>>::vector_t const & p_vec1,
   //       int const & p_distance
   //    ){
   //       trace( "[VECTOR] - Left-shifting 8 bit integer values of one register (all by the same distance) (sse)" );
   //       return _mm_slli_epi16(p_vec1, p_distance);
   //    }
   // };
   //not tested
   template<>
   struct shift_left_individual<sse<v128<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint8_t>>::vector_t
      apply(
         typename sse<v128<uint8_t>>::vector_t const & p_data,
         typename sse<v128<uint8_t>>::vector_t const & p_distance
      ){
         // SSE does not have an intrinsic for this.
         //is the comparison with 8 necessary?
         trace( "[VECTOR] - Left-shifting 8 bit integer values of one register (each by its individual distance) (sse)" );
         uint8_t distance0 = _mm_extract_epi8(p_distance, 0);
         uint8_t distance1 = _mm_extract_epi8(p_distance, 1);
         uint8_t distance2 = _mm_extract_epi8(p_distance, 2);
         uint8_t distance3 = _mm_extract_epi8(p_distance, 3);
         uint8_t distance4 = _mm_extract_epi8(p_distance, 4);
         uint8_t distance5 = _mm_extract_epi8(p_distance, 5);
         uint8_t distance6 = _mm_extract_epi8(p_distance, 6);
         uint8_t distance7 = _mm_extract_epi8(p_distance, 7);
         uint8_t distance8 = _mm_extract_epi8(p_distance, 8);
         uint8_t distance9 = _mm_extract_epi8(p_distance, 9);
         uint8_t distance10 = _mm_extract_epi8(p_distance, 10);
         uint8_t distance11 = _mm_extract_epi8(p_distance, 11);
         uint8_t distance12 = _mm_extract_epi8(p_distance, 12);
         uint8_t distance13 = _mm_extract_epi8(p_distance, 13);
         uint8_t distance14 = _mm_extract_epi8(p_distance, 14);
         uint8_t distance15 = _mm_extract_epi8(p_distance, 15);
         return _mm_set_epi8(
                 (distance15 == 8) ? 0 : (_mm_extract_epi8(p_data, 15) << distance15),
                 (distance14 == 8) ? 0 : (_mm_extract_epi8(p_data, 14) << distance14),
                 (distance13 == 8) ? 0 : (_mm_extract_epi8(p_data, 13) << distance13),
                 (distance12 == 8) ? 0 : (_mm_extract_epi8(p_data, 12) << distance12),
                 (distance11 == 8) ? 0 : (_mm_extract_epi8(p_data, 11) << distance11),
                 (distance10 == 8) ? 0 : (_mm_extract_epi8(p_data, 10) << distance10),
                 (distance9 == 8) ? 0 : (_mm_extract_epi8(p_data, 9) << distance9),
                 (distance8 == 8) ? 0 : (_mm_extract_epi8(p_data, 8) << distance8),
                 (distance7 == 8) ? 0 : (_mm_extract_epi8(p_data, 7) << distance7),
                 (distance6 == 8) ? 0 : (_mm_extract_epi8(p_data, 6) << distance6),
                 (distance5 == 8) ? 0 : (_mm_extract_epi8(p_data, 5) << distance5),
                 (distance4 == 8) ? 0 : (_mm_extract_epi8(p_data, 4) << distance4),
                 (distance3 == 8) ? 0 : (_mm_extract_epi8(p_data, 3) << distance3),
                 (distance2 == 8) ? 0 : (_mm_extract_epi8(p_data, 2) << distance2),
                 (distance1 == 8) ? 0 : (_mm_extract_epi8(p_data, 1) << distance1),
                 (distance0 == 8) ? 0 : (_mm_extract_epi8(p_data, 0) << distance0)

         );
      }
   };
   //doesn't work, no easy 8bit intrinsic to replace _mm_srli_epi16
   // template<>
   // struct shift_right<sse<v128<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename sse<v128<uint8_t>>::vector_t
   //    apply(
   //       typename sse<v128<uint8_t>>::vector_t const & p_vec1,
   //       int const & p_distance
   //    ){
   //       trace( "[VECTOR] - Right-shifting 8 bit integer values of one register (all by the same distance) (sse)" );
   //       return _mm_srli_epi16(p_vec1, p_distance);
   //    }
   // };
   // //not tested
   template<>
   struct shift_right_individual<sse<v128<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename sse<v128<uint8_t>>::vector_t
      apply(
         typename sse<v128<uint8_t>>::vector_t const & p_data,
         typename sse<v128<uint8_t>>::vector_t const & p_distance
      ){
         // SSE does not have an intrinsic for this.
         // is the comparison with 8 necessary?
         // The static_cast to an unsigned type is necessary, since the scalar
         // shift shifts in sign-bits otherwise.
         trace( "[VECTOR] - Right-shifting 16 bit integer values of one register (each by its individual distance) (sse)" );
         uint8_t distance0 = _mm_extract_epi8(p_distance, 0);
         uint8_t distance1 = _mm_extract_epi8(p_distance, 1);
         uint8_t distance2 = _mm_extract_epi8(p_distance, 2);
         uint8_t distance3 = _mm_extract_epi8(p_distance, 3);
         uint8_t distance4 = _mm_extract_epi8(p_distance, 4);
         uint8_t distance5 = _mm_extract_epi8(p_distance, 5);
         uint8_t distance6 = _mm_extract_epi8(p_distance, 6);
         uint8_t distance7 = _mm_extract_epi8(p_distance, 7);
         uint8_t distance8 = _mm_extract_epi8(p_distance, 8);
         uint8_t distance9 = _mm_extract_epi8(p_distance, 9);
         uint8_t distance10 = _mm_extract_epi8(p_distance, 10);
         uint8_t distance11 = _mm_extract_epi8(p_distance, 11);
         uint8_t distance12 = _mm_extract_epi8(p_distance, 12);
         uint8_t distance13 = _mm_extract_epi8(p_distance, 13);
         uint8_t distance14 = _mm_extract_epi8(p_distance, 14);
         uint8_t distance15 = _mm_extract_epi8(p_distance, 15);
         return _mm_set_epi8(
                 (distance15 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 15)) << distance15),
                 (distance14 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 14)) << distance14),
                 (distance13 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 13)) << distance13),
                 (distance12 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 12)) << distance12),
                 (distance11 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 11)) << distance11),
                 (distance10 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 10)) << distance10),
                 (distance9 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 9)) << distance9),
                 (distance8 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 8)) << distance8),
                 (distance7 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 7)) << distance7),
                 (distance6 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 6)) << distance6),
                 (distance5 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 5)) << distance5),
                 (distance4 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 4)) << distance4),
                 (distance3 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 3)) << distance3),
                 (distance2 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 2)) << distance2),
                 (distance1 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 1)) << distance1),
                 (distance0 == 8) ? 0 : (static_cast<uint8_t>(_mm_extract_epi8(p_data, 0)) << distance0)
         );
      }
   };

}
#endif /* MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_CALC_SSE_H */

