/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   create_sse.h
 * Author: Annett
 *
 * Created on 25. April 2019, 11:10
 */

#ifndef CREATE_SSE_H
#define CREATE_SSE_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/sse/extension_sse.h>
#include <vector/primitives/create.h>

#include <functional>

namespace vector {
    
    
   template<typename T>
   struct create<sse<v128<T>>,64> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      set( int a1, int a0) {
         trace( "[VECTOR] - set sse register." );
         return _mm_set_epi64x(a1, a0);
      }
      
    
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      set1( int a0) {
         trace( "[VECTOR] - set1 sse register." );
         return _mm_set1_epi64x(a0);
      }
   
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      set_sequence( int a, int b) {
         trace( "[VECTOR] - set_sequence sse register." );
         return _mm_set_epi64x(a+b, a);
      }
      
   };
   
   template<typename T>
   struct create<sse<v128<T>>,32> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      set( int a3, int a2, int a1, int a0) {
         trace( "[VECTOR] - set sse register." );
         return _mm_set_epi32(a3, a2, a1, a0);
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      set_sequence( int a, int b) {
         trace( "[VECTOR] - set_sequence sse register." );
         return _mm_set_epi32(a+3*b,a+2*b,a+b, a);
      }
          
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      set1( int a0) {
         trace( "[VECTOR] - set1 sse register." );
         return _mm_set1_epi32(a0);
      }
   
   };
}

#endif /* CREATE_SSE_H */

