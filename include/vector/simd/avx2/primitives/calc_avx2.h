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
   
   
   template<>
   struct add<avx2<v256<uint32_t>>/*, 32*/> {
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
   struct min<avx2<v256<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint32_t>>::vector_t
      apply(
         typename avx2<v256<uint32_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - build minimum of 32 bit integer values from two registers (avx)" );
         return _mm256_blendv_epi8(p_vec2, p_vec1, _mm256_cmpgt_epi32(p_vec2, p_vec1));
      }
   };
   
   template<>
   struct sub<avx2<v256<uint32_t>>/*, 32*/> {
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
   //doesn't work yet
   // template<>
   // struct hadd<avx2<v256<uint32_t>>/*, 32*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint32_t>>::base_t
   //    apply(
   //       typename avx2<v256<uint32_t>>::vector_t const & p_vec1
   //    ){
   //       trace( "[VECTOR] - Horizontally add 32 bit integer values one register (avx2)" );
   //       __m256i tmp =
   //          _mm256_castpd_si256(  
   //             _mm256_hadd_epi32(  
   //                _mm256_castsi256_pd(p_vec1),
   //                _mm256_castsi256_pd(p_vec1)
   //             )
   //          );
   //       return _mm256_extract_epi32(tmp,0)+_mm256_extract_epi32(tmp,2);
   //    }
   // };

   template<>
   struct mul<avx2<v256<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint32_t>>::vector_t
      apply(
         typename avx2<v256<uint32_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint32_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 32 bit integer values from two registers (avx2)" );
         info( "[VECTOR] - _mm256_mul_epu32 is called " );
         return _mm256_mullo_epi32( p_vec1, p_vec2);  
      }
   };
   //doesn't work yet
   // template<>
   // struct div<avx2<v256<uint32_t>>/*, 32*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint32_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint32_t>>::vector_t const & p_vec1,
   //       typename avx2<v256<uint32_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Divide 32 bit integer values from two registers (avx2)" );
   //       __m256d divhelper = _mm256_set1_ps(0x0010000000000000);

   //       return
   //          _mm256_xor_si256(
   //             _mm256_castpd_si256(
   //                _mm256_add_ps(
   //                   _mm256_floor_ps(
   //                      _mm256_div_ps(
   //                         _mm256_castsi256_pd(p_vec1),
   //                         _mm256_castsi256_pd(p_vec2)
   //                      )
   //                   ),
   //                   divhelper
   //                )
   //             ),
   //             _mm256_castpd_si256( //_mm256_castpd_ps?
   //                divhelper
   //             )
   //          );
   //    }
   // };
   //doesn't work yet
   // template<>
   // struct mod<avx2<v256<uint32_t>>/*, 32*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint32_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint32_t>>::vector_t const & p_vec1,
   //       typename avx2<v256<uint32_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Modulo divide 32 bit integer values from two registers (avx2)" );
   //       info( "[VECTOR] - MODULO IS A WORKAROUND" );
   //       __m256d divhelper = _mm256_set1_pd(0x0010000000000000);
   //       __m256d intermediate =
   //          _mm256_add_pd(
   //             _mm256_floor_pd(
   //                _mm256_div_pd(
   //                   _mm256_castsi256_pd(p_vec1),
   //                   _mm256_castsi256_pd(p_vec2)
   //                )
   //             ),
   //             divhelper
   //          );
   //       return
   //          _mm256_sub_epi6432(
   //             p_vec1,
   //             _mm256_mullo_epi32(
   //                _mm256_xor_si256(
   //                   _mm256_castpd_si256(intermediate),
   //                   _mm256_castpd_si256(divhelper)
   //                ),
   //                p_vec2
   //             )
   //          );
   //    }
   // };
   //tested, but not sure if output is what it was supposed to be
   template<>
   struct inv<avx2<v256<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint32_t>>::vector_t
      apply(
         typename avx2<v256<uint32_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 32 bit integer values of one register (avx2)" );
         return _mm256_sub_epi32( _mm256_set1_epi32(0), p_vec1);
      }
   };
   //not tested
   template<>
   struct shift_left<avx2<v256<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint32_t>>::vector_t
      apply(
         typename avx2<v256<uint32_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 32 bit integer values of one register (all by the same distance) (avx2)" );
         return _mm256_slli_epi32(p_vec1, p_distance);
      }
   };
   //not tested
   template<>
   struct shift_left_individual<avx2<v256<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint32_t>>::vector_t
      apply(
         typename avx2<v256<uint32_t>>::vector_t const & p_data,
         typename avx2<v256<uint32_t>>::vector_t const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 32 bit integer values of one register (each by its individual distance) (avx2)" );
         return _mm256_sllv_epi32(p_data, p_distance);
      }
   };
   //not tested
   template<>
   struct shift_right<avx2<v256<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint32_t>>::vector_t
      apply(
         typename avx2<v256<uint32_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 32 bit integer values of one register (all by the same distance) (avx2)" );
         return _mm256_srli_epi32(p_vec1, p_distance);
      }
   };
   //not tested
   template<>
   struct shift_right_individual<avx2<v256<uint32_t>>/*, 32*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint32_t>>::vector_t
      apply(
         typename avx2<v256<uint32_t>>::vector_t const & p_data,
         typename avx2<v256<uint32_t>>::vector_t const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 32 bit integer values of one register (each by its individual distance) (avx2)" );
         return _mm256_srlv_epi32(p_data, p_distance);
      }
   };

   template<>
   struct add<avx2<v256<uint16_t>>/*, 16*/> {
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
   struct min<avx2<v256<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint16_t>>::vector_t
      apply(
         typename avx2<v256<uint16_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - build minimum of 16 bit integer values from two registers (avx)" );
         return _mm256_blendv_epi8(p_vec2, p_vec1, _mm256_cmpgt_epi16(p_vec2, p_vec1));
      }
   };
   
   template<>
   struct sub<avx2<v256<uint16_t>>/*, 16*/> {
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
   //doesn't work yet
   // template<>
   // struct hadd<avx2<v256<uint16_t>>/*, 16*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint16_t>>::base_t
   //    apply(
   //       typename avx2<v256<uint16_t>>::vector_t const & p_vec1
   //    ){
   //       trace( "[VECTOR] - Horizontally add 16 bit integer values one register (avx2)" );
   //       __m256i tmp =
   //          _mm256_castpd_si256(  
   //             _mm256_hadd_epi16(  
   //                _mm256_castsi256_pd(p_vec1),
   //                _mm256_castsi256_pd(p_vec1)
   //             )
   //          );
   //       return _mm256_extract_epi16(tmp,0)+_mm256_extract_epi16(tmp,2);
   //    }
   // };

   template<>
   struct mul<avx2<v256<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint16_t>>::vector_t
      apply(
         typename avx2<v256<uint16_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint16_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 16 bit integer values from two registers (avx2)" );
         info( "[VECTOR] - _mm256_mul_epu16 is called " );
         return _mm256_mullo_epi16( p_vec1, p_vec2);  
      }
   };
   //doesn't work yet
   // template<>
   // struct div<avx2<v256<uint16_t>>/*, 16*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint16_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint16_t>>::vector_t const & p_vec1,
   //       typename avx2<v256<uint16_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Divide 16 bit integer values from two registers (avx2)" );
   //       __m256d divhelper = _mm256_set1_ps(0x0010000000000000);

   //       return
   //          _mm256_xor_si256(
   //             _mm256_castpd_si256(
   //                _mm256_add_ps(
   //                   _mm256_floor_ps(
   //                      _mm256_div_ps(
   //                         _mm256_castsi256_pd(p_vec1),
   //                         _mm256_castsi256_pd(p_vec2)
   //                      )
   //                   ),
   //                   divhelper
   //                )
   //             ),
   //             _mm256_castpd_si256( //_mm256_castpd_ps?
   //                divhelper
   //             )
   //          );
   //    }
   // };
   //doesn't work yet
   // template<>
   // struct mod<avx2<v256<uint16_t>>/*, 16*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint16_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint16_t>>::vector_t const & p_vec1,
   //       typename avx2<v256<uint16_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Modulo divide 16 bit integer values from two registers (avx2)" );
   //       info( "[VECTOR] - MODULO IS A WORKAROUND" );
   //       __m256d divhelper = _mm256_set1_pd(0x0010000000000000);
   //       __m256d intermediate =
   //          _mm256_add_pd(
   //             _mm256_floor_pd(
   //                _mm256_div_pd(
   //                   _mm256_castsi256_pd(p_vec1),
   //                   _mm256_castsi256_pd(p_vec2)
   //                )
   //             ),
   //             divhelper
   //          );
   //       return
   //          _mm256_sub_epi6432(
   //             p_vec1,
   //             _mm256_mullo_epi32(
   //                _mm256_xor_si256(
   //                   _mm256_castpd_si256(intermediate),
   //                   _mm256_castpd_si256(divhelper)
   //                ),
   //                p_vec2
   //             )
   //          );
   //    }
   // };
   //tested, but not sure if output is what it was supposed to be
   template<>
   struct inv<avx2<v256<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint16_t>>::vector_t
      apply(
         typename avx2<v256<uint16_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 16 bit integer values of one register (avx2)" );
         return _mm256_sub_epi16( _mm256_set1_epi16(0), p_vec1);
      }
   };
   //not tested
   template<>
   struct shift_left<avx2<v256<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint16_t>>::vector_t
      apply(
         typename avx2<v256<uint16_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 16 bit integer values of one register (all by the same distance) (avx2)" );
         return _mm256_slli_epi16(p_vec1, p_distance);
      }
   };
   // _mm256_sllv_epi32 can be replaced with _mm256_sllv_epi16, but this intrinsic uses AVX512VL + AVX512BW
   // template<>
   // struct shift_left_individual<avx2<v256<uint16_t>>/*, 32*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint16_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint16_t>>::vector_t const & p_data,
   //       typename avx2<v256<uint16_t>>::vector_t const & p_distance
   //    ){
   //       trace( "[VECTOR] - Left-shifting 16 bit integer values of one register (each by its individual distance) (avx2)" );
   //       return _mm256_sllv_epi32(p_data, p_distance);
   //    }
   // };
   //not tested
   template<>
   struct shift_right<avx2<v256<uint16_t>>/*, 16*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint16_t>>::vector_t
      apply(
         typename avx2<v256<uint16_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 16 bit integer values of one register (all by the same distance) (avx2)" );
         return _mm256_srli_epi16(p_vec1, p_distance);
      }
   };
   // _mm256_srlv_epi32 can be replaced with _mm256_srlv_epi16, but this intrinsic uses AVX512VL + AVX512BW
   // template<>
   // struct shift_right_individual<avx2<v256<uint16_t>>/*, 16*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint16_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint16_t>>::vector_t const & p_data,
   //       typename avx2<v256<uint16_t>>::vector_t const & p_distance
   //    ){
   //       trace( "[VECTOR] - Right-shifting 16 bit integer values of one register (each by its individual distance) (avx2)" );
   //       return _mm256_srlv_epi32(p_data, p_distance);
   //    }
   // };


   template<>
   struct add<avx2<v256<uint8_t>>/*, 8*/> {
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

   template<>
   struct min<avx2<v256<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint8_t>>::vector_t
      apply(
         typename avx2<v256<uint8_t>>::vector_t const & p_vec1,
         typename avx2<v256<uint8_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - build minimum of 8 bit integer values from two registers (avx)" );
         return _mm256_blendv_epi8(p_vec2, p_vec1, _mm256_cmpgt_epi8(p_vec2, p_vec1));
      }
   };
   
   template<>
   struct sub<avx2<v256<uint8_t>>/*, 8*/> {
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
   //doesn't work yet
   // template<>
   // struct hadd<avx2<v256<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint8_t>>::base_t
   //    apply(
   //       typename avx2<v256<uint8_t>>::vector_t const & p_vec1
   //    ){
   //       trace( "[VECTOR] - Horizontally add 8 bit integer values one register (avx2)" );
   //       __m256i tmp =
   //          _mm256_castpd_si256(  
   //             _mm256_hadd_epi16(  
   //                _mm256_castsi256_pd(p_vec1),
   //                _mm256_castsi256_pd(p_vec1)
   //             )
   //          );
   //       return _mm256_extract_epi16(tmp,0)+_mm256_extract_epi16(tmp,2);
   //    }
   // };
   //doesn't work, no easy 8bit intrinsic to replace _mm256_mullo_epi16
   // template<>
   // struct mul<avx2<v256<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint8_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint8_t>>::vector_t const & p_vec1,
   //       typename avx2<v256<uint8_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Multiply 8 bit integer values from two registers (avx2)" );
   //       info( "[VECTOR] - _mm256_mul_epu16 is called " );
   //       return _mm256_mullo_epi16( p_vec1, p_vec2);  
   //    }
   // };
   //doesn't work yet
   // template<>
   // struct div<avx2<v256<uint16_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint8_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint8_t>>::vector_t const & p_vec1,
   //       typename avx2<v256<uint8_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Divide 8 bit integer values from two registers (avx2)" );
   //       __m256d divhelper = _mm256_set1_ps(0x0010000000000000);

   //       return
   //          _mm256_xor_si256(
   //             _mm256_castpd_si256(
   //                _mm256_add_ps(
   //                   _mm256_floor_ps(
   //                      _mm256_div_ps(
   //                         _mm256_castsi256_pd(p_vec1),
   //                         _mm256_castsi256_pd(p_vec2)
   //                      )
   //                   ),
   //                   divhelper
   //                )
   //             ),
   //             _mm256_castpd_si256( //_mm256_castpd_ps?
   //                divhelper
   //             )
   //          );
   //    }
   // };
   //doesn't work yet
   // template<>
   // struct mod<avx2<v256<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint16_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint8_t>>::vector_t const & p_vec1,
   //       typename avx2<v256<uint8_t>>::vector_t const & p_vec2
   //    ){
   //       trace( "[VECTOR] - Modulo divide 8 bit integer values from two registers (avx2)" );
   //       info( "[VECTOR] - MODULO IS A WORKAROUND" );
   //       __m256d divhelper = _mm256_set1_pd(0x0010000000000000);
   //       __m256d intermediate =
   //          _mm256_add_pd(
   //             _mm256_floor_pd(
   //                _mm256_div_pd(
   //                   _mm256_castsi256_pd(p_vec1),
   //                   _mm256_castsi256_pd(p_vec2)
   //                )
   //             ),
   //             divhelper
   //          );
   //       return
   //          _mm256_sub_epi6432(
   //             p_vec1,
   //             _mm256_mullo_epi32(
   //                _mm256_xor_si256(
   //                   _mm256_castpd_si256(intermediate),
   //                   _mm256_castpd_si256(divhelper)
   //                ),
   //                p_vec2
   //             )
   //          );
   //    }
   // };
   //tested, but not sure if output is what it was supposed to be
   template<>
   struct inv<avx2<v256<uint8_t>>/*, 8*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<uint8_t>>::vector_t
      apply(
         typename avx2<v256<uint8_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 8 bit integer values of one register (avx2)" );
         return _mm256_sub_epi8( _mm256_set1_epi8(0), p_vec1);
      }
   };
   //doesn't work, no easy 8bit intrinsic to replace _mm256_slli_epi16
   // template<>
   // struct shift_left<avx2<v256<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint8_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint8_t>>::vector_t const & p_vec1,
   //       int const & p_distance
   //    ){
   //       trace( "[VECTOR] - Left-shifting 8 bit integer values of one register (all by the same distance) (avx2)" );
   //       return _mm256_slli_epi16(p_vec1, p_distance);
   //    }
   // };
   //doesn't work, no easy 8bit intrinsic to replace _mm256_sllv_epi32
   // template<>
   // struct shift_left_individual<avx2<v256<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint8_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint8_t>>::vector_t const & p_data,
   //       typename avx2<v256<uint8_t>>::vector_t const & p_distance
   //    ){
   //       trace( "[VECTOR] - Left-shifting 8 bit integer values of one register (each by its individual distance) (avx2)" );
   //       return _mm256_sllv_epi32(p_data, p_distance);
   //    }
   // };
   //doesn't work, no easy 8bit intrinsic to replace _mm256_srli_epi16
   // template<>
   // struct shift_right<avx2<v256<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint8_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint8_t>>::vector_t const & p_vec1,
   //       int const & p_distance
   //    ){
   //       trace( "[VECTOR] - Right-shifting 8 bit integer values of one register (all by the same distance) (avx2)" );
   //       return _mm256_srli_epi16(p_vec1, p_distance);
   //    }
   // };
   //doesn't work, no easy 8bit intrinsic to replace _mm256_srlv_epi32
   // template<>
   // struct shift_right_individual<avx2<v256<uint8_t>>/*, 8*/> {
   //    MSV_CXX_ATTRIBUTE_FORCE_INLINE
   //    static
   //    typename avx2<v256<uint8_t>>::vector_t
   //    apply(
   //       typename avx2<v256<uint8_t>>::vector_t const & p_data,
   //       typename avx2<v256<uint8_t>>::vector_t const & p_distance
   //    ){
   //       trace( "[VECTOR] - Right-shifting 16 bit integer values of one register (each by its individual distance) (avx2)" );
   //       return _mm256_srlv_epi32(p_data, p_distance);
   //    }
   // };


}
#endif /* MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_CALC_AVX2_H */
