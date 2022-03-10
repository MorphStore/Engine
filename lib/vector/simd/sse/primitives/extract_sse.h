/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   extract_sse.h
 * Author: Annett
 *
 * Created on 25. April 2019, 14:01
 */

#ifndef MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_EXTRACT_SSE_H
#define MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_EXTRACT_SSE_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include "../extension_sse.h"
#include "../../../primitives/extract.h"

#include <functional>

namespace vectorlib {
    
    
   template<typename T>
   struct extract<sse<v128<T>>,64> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::base_t
      extract_value( sse< v128< uint64_t > >::vector_t p_vec, int idx) {
         trace( "[VECTOR] - extract value from sse register." );
         switch (idx){
             case 0: return _mm_extract_epi64(p_vec,0); break;
             case 1: return _mm_extract_epi64(p_vec,1); break;
         }
         return (typename sse< v128< U > >::base_t)0;
      }
   };
   
      template<typename T>
   struct extract<sse<v128<T>>,32> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::base_t
      extract_value( sse< v128< uint64_t > >::vector_t p_vec, int idx) {
         trace( "[VECTOR] - extract value from sse register." );
         switch (idx){
             case 0: return _mm_extract_epi32(p_vec,0); break;
             case 1: return _mm_extract_epi32(p_vec,1); break;
             case 2: return _mm_extract_epi32(p_vec,2); break;
             case 3: return _mm_extract_epi32(p_vec,3); break;
         }
         return (typename sse< v128< U > >::base_t)0;
      }
   };
}
      

#endif /* MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_EXTRACT_SSE_H */

