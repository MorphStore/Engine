/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   create_avx2.h
 * Author: Annett
 *
 * Created on 25. April 2019, 11:11
 */

#ifndef MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_CREATE_AVX2_H
#define MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_CREATE_AVX2_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/primitives/create.h>

#include <functional>

namespace vectorlib {
    
    
   template<typename T>
   struct create<avx2<v256<T>>,64> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      set( int a3, int a2, int a1, int a0) {
         trace( "[VECTOR] - set sse register." );
         return _mm256_set_epi64x(a3, a2, a1, a0);
      }
      
    
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      set_sequence( int a, int b) {
         trace( "[VECTOR] - set_sequence sse register." );
         return _mm256_set_epi64x(a+3*b,a+2*b,a+b, a);
      }
            
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      set1( int a0) {
         trace( "[VECTOR] - set1 sse register." );
         return _mm256_set1_epi64x(a0);
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2<v256< T > >::mask_t
      init0( void ) {
         return (typename avx2<v256<T>>::mask_t) 0;
      }
   
   };
   
   template<typename T>
   struct create<avx2<v256<T>>,32> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      set( int a7, int a6, int a5, int a4, int a3, int a2, int a1, int a0) {
         trace( "[VECTOR] - set sse register." );
         return _mm256_set_epi32(a7, a6, a5, a4, a3, a2, a1, a0);
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      set_sequence( int a, int b) {
         trace( "[VECTOR] - set_sequence sse register." );
         return _mm256_set_epi32(a+7*b,a+6*b,a+5*b, a+4*b,a+3*b,a+2*b,a+b, a);
      }
          
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      set1( int a0) {
         trace( "[VECTOR] - set1 sse register." );
         return _mm256_set1_epi32(a0);
      }
   
   };
}

#endif /* MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_CREATE_AVX2_H */

