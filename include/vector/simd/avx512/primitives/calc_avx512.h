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

#ifndef MORPHSTORE_VECTOR_SIMD_AVX512_PRIMITIVES_CALC_AVX512_H
#define MORPHSTORE_VECTOR_SIMD_AVX512_PRIMITIVES_CALC_AVX512_H



#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx512/extension_avx512.h>
#include <vector/primitives/calc.h>

#include <functional>

namespace vectorlib{
   template<>
   struct add<avx512<v512<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::vector_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 64 bit integer values from two registers (avx512)" );
         return _mm512_add_epi64( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct add<avx512<v256<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<uint64_t>>::vector_t
      apply(
         typename avx512<v256<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v256<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 64 bit integer values from two registers (avx512)" );
         return _mm256_add_epi64( p_vec1, p_vec2);
      }
   };
   template<>
   struct add<avx512<v128<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<uint64_t>>::vector_t
      apply(
         typename avx512<v128<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v128<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 64 bit integer values from two registers (avx512)" );
         return _mm_add_epi64( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct min<avx512<v512<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::vector_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Minimum of 64 bit integer values from two registers (avx512)" );
         return _mm512_min_epi64( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct min<avx512<v256<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<uint64_t>>::vector_t
      apply(
         typename avx512<v256<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v256<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 64 bit integer values from two registers (avx512)" );
         return _mm256_min_epi64( p_vec1, p_vec2);
      }
   };
   template<>
   struct min<avx512<v128<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<uint64_t>>::vector_t
      apply(
         typename avx512<v128<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v128<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 64 bit integer values from two registers (avx512)" );
         return _mm_min_epi64( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct sub<avx512<v512<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::vector_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 64 bit integer values from two registers (avx512)" );
         return _mm512_sub_epi64( p_vec1, p_vec2);
      }
   };
   template<>
   struct hadd<avx512<v512<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::base_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Horizontally add 64 bit integer values one register (avx512)" );
         return _mm512_reduce_add_epi64(p_vec1);
      }
   };
   template<>
   struct mul<avx512<v512<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::vector_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (avx512)" );
         info( "[VECTOR] - _mm512_mul_epu32 is called (only the lower 32 bit are actually processed" );
         return _mm512_mul_epu32( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct mul<avx512<v256<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<uint64_t>>::vector_t
      apply(
         typename avx512<v256<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v256<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (avx512)" );
         info( "[VECTOR] - _mm256_mul_epu32 is called (only the lower 32 bit are actually processed" );
         return _mm256_mul_epu32( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct mul<avx512<v128<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<uint64_t>>::vector_t
      apply(
         typename avx512<v128<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v128<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (avx512)" );
         info( "[VECTOR] - _mm_mul_epu32 is called (only the lower 32 bit are actually processed" );
         return _mm_mul_epu32( p_vec1, p_vec2);
      }
   };
   
   
   template<>
   struct div<avx512<v512<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::vector_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Divide 64 bit integer values from two registers (avx512)" );
         __m512d divhelper = _mm512_set1_pd(0x0010000000000000);
         return
            _mm512_xor_si512(
               _mm512_castpd_si512(
                  _mm512_add_pd(
                     // @todo This rounds the result to the nearest integer,
                     // but we want it to be rounded down, since this would be
                     // the expected outcome of an integer division. There is
                     // no _mm512_floor_pd (like in SSE and AVX). I tried
                     // _mm512_div_round_pd with all possible rounding modes,
                     // but none of them worked...
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
   struct mod<avx512<v512<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::vector_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Modulo divide 64 bit integer values from two registers (avx512)" );
         info( "[VECTOR] - MODULO IS A WORKAROUND" );
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
   struct inv<avx512<v512<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::vector_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 64 bit integer values of one register (avx512)" );
         return _mm512_sub_epi64( _mm512_set1_epi64(0), p_vec1);
      }
   };
   template<>
   struct shift_left<avx512<v512<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::vector_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (all by the same distance) (avx512)" );
         return _mm512_slli_epi64(p_vec1, p_distance);
      }
   };
   template<>
   struct shift_left_individual<avx512<v512<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::vector_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_data,
         typename avx512<v512<uint64_t>>::vector_t const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (each by its individual distance) (avx512)" );
         return _mm512_sllv_epi64(p_data, p_distance);
      }
   };
   template<>
   struct shift_right<avx512<v512<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::vector_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (all by the same distance) (avx512)" );
         return _mm512_srli_epi64(p_vec1, p_distance);
      }
   };
   template<>
   struct shift_right_individual<avx512<v512<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::vector_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_data,
         typename avx512<v512<uint64_t>>::vector_t const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (each by its individual distance) (avx512)" );
         return _mm512_srlv_epi64(p_data, p_distance);
      }
   };

   
//all of the functions below are not tested
   template<>
   struct add<avx512<v512<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint32_t>>::vector_t
      apply(
         typename avx512<v512<uint32_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 32 bit integer values from two registers (avx512)" );
         return _mm512_add_epi32( p_vec1, p_vec2);
      }
   };
   template<>
   struct add<avx512<v256<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<uint32_t>>::vector_t
      apply(
         typename avx512<v256<uint32_t>>::vector_t const & p_vec1,
         typename avx512<v256<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 32 bit integer values from two registers (avx512)" );
         return _mm256_add_epi32( p_vec1, p_vec2);
      }
   };
   template<>
   struct add<avx512<v128<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<uint32_t>>::vector_t
      apply(
         typename avx512<v128<uint32_t>>::vector_t const & p_vec1,
         typename avx512<v128<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 32 bit integer values from two registers (avx512)" );
         return _mm_add_epi32( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct min<avx512<v512<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint32_t>>::vector_t
      apply(
         typename avx512<v512<uint32_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Minimum of 32 bit integer values from two registers (avx512)" );
         return _mm512_min_epi32( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct min<avx512<v256<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<uint32_t>>::vector_t
      apply(
         typename avx512<v256<uint32_t>>::vector_t const & p_vec1,
         typename avx512<v256<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 32 bit integer values from two registers (avx512)" );
         return _mm256_min_epi32( p_vec1, p_vec2);
      }
   };
   template<>
   struct min<avx512<v128<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<uint32_t>>::vector_t
      apply(
         typename avx512<v128<uint32_t>>::vector_t const & p_vec1,
         typename avx512<v128<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 32 bit integer values from two registers (avx512)" );
         return _mm_min_epi32( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct sub<avx512<v512<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint32_t>>::vector_t
      apply(
         typename avx512<v512<uint32_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 32 bit integer values from two registers (avx512)" );
         return _mm512_sub_epi32( p_vec1, p_vec2);
      }
   };
   template<>
   struct hadd<avx512<v512<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint32_t>>::base_t
      apply(
         typename avx512<v512<uint32_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Horizontally add 32 bit integer values one register (avx512)" );
         return _mm512_reduce_add_epi32(p_vec1);
      }
   };
   template<>
   struct mul<avx512<v512<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint32_t>>::vector_t
      apply(
         typename avx512<v512<uint32_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 32 bit integer values from two registers (avx512)" );
         info( "[VECTOR] - _mm512_mullo_epi32 is called" );
         return _mm512_mullo_epi32( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct mul<avx512<v256<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<uint32_t>>::vector_t
      apply(
         typename avx512<v256<uint32_t>>::vector_t const & p_vec1,
         typename avx512<v256<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 32 bit integer values from two registers (avx512)" );
         info( "[VECTOR] - _mm256_mullo_epi32 is called" );
         return _mm256_mullo_epi32( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct mul<avx512<v128<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<uint32_t>>::vector_t
      apply(
         typename avx512<v128<uint32_t>>::vector_t const & p_vec1,
         typename avx512<v128<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 32 bit integer values from two registers (avx512)" );
         info( "[VECTOR] - _mm_mullo_epi32 is called (only the lower 32 bit are actually processed" );
         return _mm_mullo_epi32( p_vec1, p_vec2);
      }
   };
   
   //doesn't work yet
   // template<>
   // struct div<avx512<v512<uint32_t>>/*, 32*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint32_t>>::vector_t
   //    apply(
   //       typename avx512<v512<uint32_t>>::vector_t const & p_vec1,
   //       typename avx512<v512<uint32_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Divide 32 bit integer values from two registers (avx512)" );
   //       __m512d divhelper = _mm512_set1_pd(0x0010000000000000);
   //       return
   //          _mm512_xor_si512(
   //             _mm512_castpd_si512(
   //                _mm512_add_pd(
   //                   // @todo This rounds the result to the nearest integer,
   //                   // but we want it to be rounded down, since this would be
   //                   // the expected outcome of an integer division. There is
   //                   // no _mm512_floor_pd (like in SSE and AVX). I tried
   //                   // _mm512_div_round_pd with all possible rounding modes,
   //                   // but none of them worked...
   //                   _mm512_div_pd(
   //                      _mm512_castsi512_pd(p_vec1),
   //                      _mm512_castsi512_pd(p_vec2)
   //                   ),
   //                   divhelper
   //                )
   //             ),
   //             _mm512_castpd_si512(
   //                divhelper
   //             )
   //          );
   //    }
   // };
   //doesn't work yet
   // template<>
   // struct mod<avx512<v512<uint32_t>>/*, 32*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint32_t>>::vector_t
   //    apply(
   //       typename avx512<v512<uint32_t>>::vector_t const & p_vec1,
   //       typename avx512<v512<uint32_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Modulo divide 32 bit integer values from two registers (avx512)" );
   //       info( "[VECTOR] - MODULO IS A WORKAROUND" );
   //       __m512d divhelper = _mm512_set1_pd(0x0010000000000000);
   //       __m512d intermediate =
   //          _mm512_add_pd(
   //             _mm512_floor_pd(
   //                _mm512_div_pd(
   //                   _mm512_castsi512_pd(p_vec1),
   //                   _mm512_castsi512_pd(p_vec2)
   //                )
   //             ),
   //             divhelper
   //          );
   //       return
   //          _mm512_sub_epi32(
   //             p_vec1,
   //             _mm512_mul_epi32(
   //                _mm512_xor_si512(
   //                   _mm512_castpd_si512(intermediate),
   //                   _mm512_castpd_si512(divhelper)
   //                ),
   //                p_vec2
   //             )
   //          );
   //    }
   // };
   template<>
   struct inv<avx512<v512<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint32_t>>::vector_t
      apply(
         typename avx512<v512<uint32_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 32 bit integer values of one register (avx512)" );
         return _mm512_sub_epi32( _mm512_set1_epi32(0), p_vec1);
      }
   };
   template<>
   struct shift_left<avx512<v512<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint32_t>>::vector_t
      apply(
         typename avx512<v512<uint32_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 32 bit integer values of one register (all by the same distance) (avx512)" );
         return _mm512_slli_epi32(p_vec1, p_distance);
      }
   };
   template<>
   struct shift_left_individual<avx512<v512<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint32_t>>::vector_t
      apply(
         typename avx512<v512<uint32_t>>::vector_t const & p_data,
         typename avx512<v512<uint32_t>>::vector_t const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 32 bit integer values of one register (each by its individual distance) (avx512)" );
         return _mm512_sllv_epi32(p_data, p_distance);
      }
   };
   template<>
   struct shift_right<avx512<v512<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint32_t>>::vector_t
      apply(
         typename avx512<v512<uint32_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 32 bit integer values of one register (all by the same distance) (avx512)" );
         return _mm512_srli_epi32(p_vec1, p_distance);
      }
   };
   template<>
   struct shift_right_individual<avx512<v512<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint32_t>>::vector_t
      apply(
         typename avx512<v512<uint32_t>>::vector_t const & p_data,
         typename avx512<v512<uint32_t>>::vector_t const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 32 bit integer values of one register (each by its individual distance) (avx512)" );
         return _mm512_srlv_epi32(p_data, p_distance);
      }
   };
   
   template<>
   struct add<avx512<v512<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint16_t>>::vector_t
      apply(
         typename avx512<v512<uint16_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 16 bit integer values from two registers (avx512)" );
         return _mm512_add_epi16( p_vec1, p_vec2);
      }
   };
   template<>
   struct add<avx512<v256<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<uint16_t>>::vector_t
      apply(
         typename avx512<v256<uint16_t>>::vector_t const & p_vec1,
         typename avx512<v256<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 16 bit integer values from two registers (avx512)" );
         return _mm256_add_epi16( p_vec1, p_vec2);
      }
   };
   template<>
   struct add<avx512<v128<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<uint16_t>>::vector_t
      apply(
         typename avx512<v128<uint16_t>>::vector_t const & p_vec1,
         typename avx512<v128<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 16 bit integer values from two registers (avx512)" );
         return _mm_add_epi16( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct min<avx512<v512<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint16_t>>::vector_t
      apply(
         typename avx512<v512<uint16_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Minimum of 16 bit integer values from two registers (avx512)" );
         return _mm512_min_epi16( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct min<avx512<v256<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<uint16_t>>::vector_t
      apply(
         typename avx512<v256<uint16_t>>::vector_t const & p_vec1,
         typename avx512<v256<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 16 bit integer values from two registers (avx512)" );
         return _mm256_min_epi16( p_vec1, p_vec2);
      }
   };
   template<>
   struct min<avx512<v128<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<uint16_t>>::vector_t
      apply(
         typename avx512<v128<uint16_t>>::vector_t const & p_vec1,
         typename avx512<v128<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 16 bit integer values from two registers (avx512)" );
         return _mm_min_epi16( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct sub<avx512<v512<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint16_t>>::vector_t
      apply(
         typename avx512<v512<uint16_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 16 bit integer values from two registers (avx512)" );
         return _mm512_sub_epi16( p_vec1, p_vec2);
      }
   };
   //doesn't work, no easy 16bit intrinsic to replace _mm512_reduce_add_epi32
   // template<>
   // struct hadd<avx512<v512<uint16_t>>/*, 16*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint16_t>>::base_t
   //    apply(
   //       typename avx512<v512<uint16_t>>::vector_t const & p_vec1
   //    ){
   //       trace( "[VECTOR] - Horizontally add 16 bit integer values one register (avx512)" );
   //       return _mm512_reduce_add_epi32(p_vec1);
   //    }
   // };
   template<>
   struct mul<avx512<v512<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint16_t>>::vector_t
      apply(
         typename avx512<v512<uint16_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 16 bit integer values from two registers (avx512)" );
         info( "[VECTOR] - _mm512_mullo_epi16 is called" );
         return _mm512_mullo_epi16( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct mul<avx512<v256<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<uint16_t>>::vector_t
      apply(
         typename avx512<v256<uint16_t>>::vector_t const & p_vec1,
         typename avx512<v256<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 16 bit integer values from two registers (avx512)" );
         info( "[VECTOR] - _mm256_mullo_epi16 is called" );
         return _mm256_mullo_epi16( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct mul<avx512<v128<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<uint16_t>>::vector_t
      apply(
         typename avx512<v128<uint16_t>>::vector_t const & p_vec1,
         typename avx512<v128<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 16 bit integer values from two registers (avx512)" );
         info( "[VECTOR] - _mm_mullo_epi16 is called" );
         return _mm_mullo_epi16( p_vec1, p_vec2);
      }
   };
   
   //doesn't work yet
   // template<>
   // struct div<avx512<v512<uint16_t>>/*, 16*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint16_t>>::vector_t
   //    apply(
   //       typename avx512<v512<uint16_t>>::vector_t const & p_vec1,
   //       typename avx512<v512<uint16_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Divide 16 bit integer values from two registers (avx512)" );
   //       __m512d divhelper = _mm512_set1_pd(0x0010000000000000);
   //       return
   //          _mm512_xor_si512(
   //             _mm512_castpd_si512(
   //                _mm512_add_pd(
   //                   // @todo This rounds the result to the nearest integer,
   //                   // but we want it to be rounded down, since this would be
   //                   // the expected outcome of an integer division. There is
   //                   // no _mm512_floor_pd (like in SSE and AVX). I tried
   //                   // _mm512_div_round_pd with all possible rounding modes,
   //                   // but none of them worked...
   //                   _mm512_div_pd(
   //                      _mm512_castsi512_pd(p_vec1),
   //                      _mm512_castsi512_pd(p_vec2)
   //                   ),
   //                   divhelper
   //                )
   //             ),
   //             _mm512_castpd_si512(
   //                divhelper
   //             )
   //          );
   //    }
   // };
   //doesn't work yet
   // template<>
   // struct mod<avx512<v512<uint16_t>>/*, 16*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint16_t>>::vector_t
   //    apply(
   //       typename avx512<v512<uint16_t>>::vector_t const & p_vec1,
   //       typename avx512<v512<uint16_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Modulo divide 16 bit integer values from two registers (avx512)" );
   //       info( "[VECTOR] - MODULO IS A WORKAROUND" );
   //       __m512d divhelper = _mm512_set1_pd(0x0010000000000000);
   //       __m512d intermediate =
   //          _mm512_add_pd(
   //             _mm512_floor_pd(
   //                _mm512_div_pd(
   //                   _mm512_castsi512_pd(p_vec1),
   //                   _mm512_castsi512_pd(p_vec2)
   //                )
   //             ),
   //             divhelper
   //          );
   //       return
   //          _mm512_sub_epi32(
   //             p_vec1,
   //             _mm512_mul_epi32(
   //                _mm512_xor_si512(
   //                   _mm512_castpd_si512(intermediate),
   //                   _mm512_castpd_si512(divhelper)
   //                ),
   //                p_vec2
   //             )
   //          );
   //    }
   // };
   template<>
   struct inv<avx512<v512<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint16_t>>::vector_t
      apply(
         typename avx512<v512<uint16_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 16 bit integer values of one register (avx512)" );
         return _mm512_sub_epi16( _mm512_set1_epi16(0), p_vec1);
      }
   };
   template<>
   struct shift_left<avx512<v512<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint16_t>>::vector_t
      apply(
         typename avx512<v512<uint16_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 16 bit integer values of one register (all by the same distance) (avx512)" );
         return _mm512_slli_epi16(p_vec1, p_distance);
      }
   };
   template<>
   struct shift_left_individual<avx512<v512<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint16_t>>::vector_t
      apply(
         typename avx512<v512<uint16_t>>::vector_t const & p_data,
         typename avx512<v512<uint16_t>>::vector_t const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 16 bit integer values of one register (each by its individual distance) (avx512)" );
         return _mm512_sllv_epi16(p_data, p_distance);
      }
   };
   template<>
   struct shift_right<avx512<v512<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint16_t>>::vector_t
      apply(
         typename avx512<v512<uint16_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 16 bit integer values of one register (all by the same distance) (avx512)" );
         return _mm512_srli_epi16(p_vec1, p_distance);
      }
   };
   template<>
   struct shift_right_individual<avx512<v512<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint16_t>>::vector_t
      apply(
         typename avx512<v512<uint16_t>>::vector_t const & p_data,
         typename avx512<v512<uint16_t>>::vector_t const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 16 bit integer values of one register (each by its individual distance) (avx512)" );
         return _mm512_srlv_epi16(p_data, p_distance);
      }
   };

   template<>
   struct add<avx512<v512<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint8_t>>::vector_t
      apply(
         typename avx512<v512<uint8_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 8 bit integer values from two registers (avx512)" );
         return _mm512_add_epi8( p_vec1, p_vec2);
      }
   };
   template<>
   struct add<avx512<v256<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<uint8_t>>::vector_t
      apply(
         typename avx512<v256<uint8_t>>::vector_t const & p_vec1,
         typename avx512<v256<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 8 bit integer values from two registers (avx512)" );
         return _mm256_add_epi8( p_vec1, p_vec2);
      }
   };
   template<>
   struct add<avx512<v128<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<uint8_t>>::vector_t
      apply(
         typename avx512<v128<uint8_t>>::vector_t const & p_vec1,
         typename avx512<v128<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 8 bit integer values from two registers (avx512)" );
         return _mm_add_epi8( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct min<avx512<v512<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint8_t>>::vector_t
      apply(
         typename avx512<v512<uint8_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Minimum of 8 bit integer values from two registers (avx512)" );
         return _mm512_min_epi8( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct min<avx512<v256<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<uint8_t>>::vector_t
      apply(
         typename avx512<v256<uint8_t>>::vector_t const & p_vec1,
         typename avx512<v256<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 8 bit integer values from two registers (avx512)" );
         return _mm256_min_epi8( p_vec1, p_vec2);
      }
   };
   template<>
   struct min<avx512<v128<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<uint8_t>>::vector_t
      apply(
         typename avx512<v128<uint8_t>>::vector_t const & p_vec1,
         typename avx512<v128<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 8 bit integer values from two registers (avx512)" );
         return _mm_min_epi8( p_vec1, p_vec2);
      }
   };
   
   template<>
   struct sub<avx512<v512<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint8_t>>::vector_t
      apply(
         typename avx512<v512<uint8_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 8 bit integer values from two registers (avx512)" );
         return _mm512_sub_epi8( p_vec1, p_vec2);
      }
   };
   //doesn't work, no easy 8bit intrinsic to replace _mm512_reduce_add_epi32
   // template<>
   // struct hadd<avx512<v512<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint8_t>>::base_t
   //    apply(
   //       typename avx512<v512<uint8_t>>::vector_t const & p_vec1
   //    ){
   //       trace( "[VECTOR] - Horizontally add 8 bit integer values one register (avx512)" );
   //       return _mm512_reduce_add_epi32(p_vec1);
   //    }
   // };
   //doesn't work, no easy 8bit intrinsic to replace _mm512_mullo_epi16
   // template<>
   // struct mul<avx512<v512<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint8_t>>::vector_t
   //    apply(
   //       typename avx512<v512<uint8_t>>::vector_t const & p_vec1,
   //       typename avx512<v512<uint8_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Multiply 8 bit integer values from two registers (avx512)" );
   //       info( "[VECTOR] - _mm512_mullo_epi16 is called" );
   //       return _mm512_mullo_epi16( p_vec1, p_vec2);
   //    }
   // };
   //doesn't work yet
   // template<>
   // struct div<avx512<v512<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint8_t>>::vector_t
   //    apply(
   //       typename avx512<v512<uint8_t>>::vector_t const & p_vec1,
   //       typename avx512<v512<uint8_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Divide 8 bit integer values from two registers (avx512)" );
   //       __m512d divhelper = _mm512_set1_pd(0x0010000000000000);
   //       return
   //          _mm512_xor_si512(
   //             _mm512_castpd_si512(
   //                _mm512_add_pd(
   //                   // @todo This rounds the result to the nearest integer,
   //                   // but we want it to be rounded down, since this would be
   //                   // the expected outcome of an integer division. There is
   //                   // no _mm512_floor_pd (like in SSE and AVX). I tried
   //                   // _mm512_div_round_pd with all possible rounding modes,
   //                   // but none of them worked...
   //                   _mm512_div_pd(
   //                      _mm512_castsi512_pd(p_vec1),
   //                      _mm512_castsi512_pd(p_vec2)
   //                   ),
   //                   divhelper
   //                )
   //             ),
   //             _mm512_castpd_si512(
   //                divhelper
   //             )
   //          );
   //    }
   // };
   //doesn't work yet
   // template<>
   // struct mod<avx512<v512<uint16_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint8_t>>::vector_t
   //    apply(
   //       typename avx512<v512<uint8_t>>::vector_t const & p_vec1,
   //       typename avx512<v512<uint8_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Modulo divide 8 bit integer values from two registers (avx512)" );
   //       info( "[VECTOR] - MODULO IS A WORKAROUND" );
   //       __m512d divhelper = _mm512_set1_pd(0x0010000000000000);
   //       __m512d intermediate =
   //          _mm512_add_pd(
   //             _mm512_floor_pd(
   //                _mm512_div_pd(
   //                   _mm512_castsi512_pd(p_vec1),
   //                   _mm512_castsi512_pd(p_vec2)
   //                )
   //             ),
   //             divhelper
   //          );
   //       return
   //          _mm512_sub_epi32(
   //             p_vec1,
   //             _mm512_mul_epi32(
   //                _mm512_xor_si512(
   //                   _mm512_castpd_si512(intermediate),
   //                   _mm512_castpd_si512(divhelper)
   //                ),
   //                p_vec2
   //             )
   //          );
   //    }
   // };
   template<>
   struct inv<avx512<v512<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint8_t>>::vector_t
      apply(
         typename avx512<v512<uint8_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 8 bit integer values of one register (avx512)" );
         return _mm512_sub_epi8( _mm512_set1_epi8(0), p_vec1);
      }
   };
   //doesn't work, no easy 8bit intrinsic to replace _mm512_slli_epi16
   // template<>
   // struct shift_left<avx512<v512<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint8_t>>::vector_t
   //    apply(
   //       typename avx512<v512<uint8_t>>::vector_t const & p_vec1,
   //       int const & p_distance
   //    ){
   //       trace( "[VECTOR] - Left-shifting 8 bit integer values of one register (all by the same distance) (avx512)" );
   //       return _mm512_slli_epi16(p_vec1, p_distance);
   //    }
   // };
   //doesn't work, no easy 8bit intrinsic to replace _mm512_sllv_epi16
   // template<>
   // struct shift_left_individual<avx512<v512<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint8_t>>::vector_t
   //    apply(
   //       typename avx512<v512<uint8_t>>::vector_t const & p_data,
   //       typename avx512<v512<uint8_t>>::vector_t const & p_distance
   //    ){
   //       trace( "[VECTOR] - Left-shifting 8 bit integer values of one register (each by its individual distance) (avx512)" );
   //       return _mm512_sllv_epi16(p_data, p_distance);
   //    }
   // };
   //doesn't work, no easy 8bit intrinsic to replace _mm512_srli_epi16
   // template<>
   // struct shift_right<avx512<v512<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint8_t>>::vector_t
   //    apply(
   //       typename avx512<v512<uint8_t>>::vector_t const & p_vec1,
   //       int const & p_distance
   //    ){
   //       trace( "[VECTOR] - Right-shifting 8 bit integer values of one register (all by the same distance) (avx512)" );
   //       return _mm512_srli_epi16(p_vec1, p_distance);
   //    }
   // };
   //doesn't work, no easy 8bit intrinsic to replace _mm512_srlv_epi16
   // template<>
   // struct shift_right_individual<avx512<v512<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx512<v512<uint8_t>>::vector_t
   //    apply(
   //       typename avx512<v512<uint8_t>>::vector_t const & p_data,
   //       typename avx512<v512<uint8_t>>::vector_t const & p_distance
   //    ){
   //       trace( "[VECTOR] - Right-shifting 8 bit integer values of one register (each by its individual distance) (avx512)" );
   //       return _mm512_srlv_epi16(p_data, p_distance);
   //    }
   // };
}
#endif /* MORPHSTORE_VECTOR_SIMD_AVX512_PRIMITIVES_CALC_AVX512_H */

