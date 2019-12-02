/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   create_avx512.h
 * Author: Annett
 *
 * Created on 25. April 2019, 11:11
 */

#ifndef MORPHSTORE_VECTOR_SIMD_AVX512_PRIMITIVES_CREATE_AVX512_H
#define MORPHSTORE_VECTOR_SIMD_AVX512_PRIMITIVES_CREATE_AVX512_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx512/extension_avx512.h>
#include <vector/primitives/create.h>

#include <functional>

namespace vectorlib {
    
    
   template<typename T>
   struct create<avx512<v512<T>>,64> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v512< U > >::vector_t
      set( int a7, int a6, int a5, int a4, int a3, int a2, int a1, int a0) {
         trace( "[VECTOR] - set sse register." );
         return _mm512_set_epi64(a7, a6, a5, a4,a3, a2, a1, a0);
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v512< U > >::vector_t
      set_sequence( int a, int b) {
         trace( "[VECTOR] - set_sequence sse register." );
         return _mm512_set_epi64(a+7*b,a+6*b,a+5*b, a+4*b,a+3*b,a+2*b,a+b, a);
      }
          
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v512< U > >::vector_t
      set1( int a0) {
         trace( "[VECTOR] - set1 sse register." );
         return _mm512_set1_epi64(a0);
      }
   
   };
   
   template<typename T>
   struct create<avx512<v512<T>>,32> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v512< U > >::vector_t
      set( int a15, int a14, int a13, int a12, int a11, int a10, int a9, int a8,int a7, int a6, int a5, int a4, int a3, int a2, int a1, int a0) {
         trace( "[VECTOR] - set sse register." );
         return _mm512_set_epi32(a15, a14, a13, a12, a11, a10, a9, a8,a7, a6, a5, a4, a3, a2, a1, a0);
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v512< U > >::vector_t
      set_sequence( int a, int b) {
         trace( "[VECTOR] - set_sequence sse register." );
         return _mm512_set_epi32(a+15*b,a+14*b,a+13*b, a+12*b,a+11*b,a+10*b,a+9*b, a+8*b,a+7*b,a+6*b,a+5*b, a+4*b,a+3*b,a+2*b,a+b, a);
      }
          
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v512< U > >::vector_t
      set1( int a0) {
         trace( "[VECTOR] - set1 sse register." );
         return _mm512_set1_epi32(a0);
      }
   
   };
   //not tested
   template<typename T>
   struct create<avx512<v512<T>>,16> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v512< U > >::vector_t
      set( int a31,int a30, int a29, int a28, int a27, int a26, int a25, int a24, int a23, int a22, int a21, int a20, int a19, int a18, int a17, 
         int a16, int a15, int a14, int a13, int a12, int a11, int a10, int a9, int a8, int a7, int a6, int a5, int a4, int a3, int a2, int a1, int a0) {
         trace( "[VECTOR] - set sse register." );
         return _mm512_set_epi16(a31, a30, a29, a28, a27, a26, a25, a24, a23, a22, a21, a20, a19, a18, a17, a16
            a15, a14, a13, a12, a11, a10, a9, a8,a7, a6, a5, a4, a3, a2, a1, a0);
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v512< U > >::vector_t
      set_sequence( int a, int b) {
         trace( "[VECTOR] - set_sequence sse register." );
         return _mm512_set_epi16(a+31*b,a+30*b,a+29*b,a+28*b,a+27*b,a+26*b,a+25*b,a+24*b,a+23*b,a+22*b,a+21*b,a+20*b,
            a+19*b, a+18*b,a+17*b,a+16*b,a+15*b,a+14*b,a+13*b,a+12*b,a+11*b,a+10*b,a+9*b,a+8*b,a+7*b,a+6*b,a+5*b, a+4*b,a+3*b,a+2*b,a+b, a);
      }
          
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v512< U > >::vector_t
      set1( int a0) {
         trace( "[VECTOR] - set1 sse register." );
         return _mm512_set1_epi16(a0);
      }
   
   };
   //not tested
   template<typename T>
   struct create<avx512<v512<T>>,8> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v512< U > >::vector_t
      set(int a63, int a62, int a61, int a60, int a59, int a58, int a57, int a56, int a55, int a54, int a53, int a52, int a51, int a50, int a49, int a48,
      int a47, int a46, int a45, int a44, int a43, int a42, int a41, int a40, int a39, int a38, int a37, int a36, int a35, int a34, int a33, int a32, 
      int a31,int a30, int a29, int a28, int a27, int a26, int a25, int a24, int a23, int a22, int a21, int a20, int a19, int a18, int a17, 
         int a16, int a15, int a14, int a13, int a12, int a11, int a10, int a9, int a8, int a7, int a6, int a5, int a4, int a3, int a2, int a1, int a0) {
         trace( "[VECTOR] - set sse register." );
         return _mm512_set_epi8(a63, a62, a61, a60, a59, a58, a57, a56, a55, a54, a53, a52, a51, a50, a49, a48, a47, a46, a45, a44, a43, a42, a41, a40,
            a39, a38, a37, a36, a35, a34, a33, a32, a31, a30, a29, a28, a27, a26, a25, a24, a23, a22, a21, a20, a19, a18, a17, a16
            a15, a14, a13, a12, a11, a10, a9, a8,a7, a6, a5, a4, a3, a2, a1, a0);
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v512< U > >::vector_t
      set_sequence( int a, int b) {
         trace( "[VECTOR] - set_sequence sse register." );
         return _mm512_set_epi8(a+63*b,a+62*b,a+61*b,a+60*b,a+59*b,a+58*b,a+57*b,a+56*b,a+55*b,a+54*b,a+53*b,a+52*b,a+51*b,a+50*b,
            a+49*b, a+48*b,a+47*b,a+46*b,a+45*b,a+44*b,a+43*b,a+42*b,a+41*b,a+40*b,a+39*b,a+38*b,a+37*b,a+36*b,a+35*b, a+34*b,a+33*b,a+32*b,a+31*b,a+30*b,a+29*b,a+28*b,a+27*b,a+26*b,a+25*b,a+24*b,a+23*b,a+22*b,a+21*b,a+20*b,
            a+19*b, a+18*b,a+17*b,a+16*b,a+15*b,a+14*b,a+13*b,a+12*b,a+11*b,a+10*b,a+9*b,a+8*b,a+7*b,a+6*b,a+5*b, a+4*b,a+3*b,a+2*b,a+b, a);
      }
          
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v512< U > >::vector_t
      set1( int a0) {
         trace( "[VECTOR] - set1 sse register." );
         return _mm512_set1_epi8(a0);
      }
   
   };
   
   template<typename T>
   struct create<avx512<v256<T>>,64> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v256< U > >::vector_t
      set_sequence( int a, int b) {
         trace( "[VECTOR] - set_sequence sse register." );
         return _mm256_set_epi64x(a+3*b,a+2*b,a+b, a);
      }
          
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v256< U > >::vector_t
      set1( int a0) {
         trace( "[VECTOR] - set1 sse register." );
         return _mm256_set1_epi64x(a0);
      }
   };
      
   template<typename T>
   struct create<avx512<v128<T>>,64> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v128< U > >::vector_t
      set_sequence( int a, int b) {
         trace( "[VECTOR] - set_sequence sse register." );
         return _mm_set_epi64x(a+b, a);
      }
          
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512< v128< U > >::vector_t
      set1( int a0) {
         trace( "[VECTOR] - set1 sse register." );
         return _mm_set1_epi64x(a0);
      }
   };
      
}

#endif /* MORPHSTORE_VECTOR_SIMD_AVX512_PRIMITIVES_CREATE_AVX512_H */

