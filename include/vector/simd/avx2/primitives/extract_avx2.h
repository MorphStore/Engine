/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   extract_avx2.h
 * Author: Annett
 *
 * Created on 25. April 2019, 14:00
 */

#ifndef EXTRACT_AVX2_H
#define EXTRACT_AVX2_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/primitives/extract.h>

#include <functional>

namespace vector {
    
    
   template<typename T>
   struct extract<avx2<v256<T>>,64> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::base_t
      extract_value( avx2< v256< uint64_t > >::vector_t p_vec, int idx) {
         trace( "[VECTOR] - extract value from sse register." );
         switch (idx){
             case 0: return _mm256_extract_epi64(p_vec,0); break;
             case 1: return _mm256_extract_epi64(p_vec,1); break;
             case 2: return _mm256_extract_epi64(p_vec,2); break;
             case 3: return _mm256_extract_epi64(p_vec,3); break;
         }
         return (typename avx2< v256< U > >::base_t)0;
      }
   };
   
      template<typename T>
   struct extract<avx2<v256<T>>,32> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::base_t
      extract_value( avx2< v256< uint64_t > >::vector_t p_vec, int idx) {
         trace( "[VECTOR] - extract value from sse register." );
         switch (idx){
             case 0: return _mm256_extract_epi32(p_vec,0); break;
             case 1: return _mm256_extract_epi32(p_vec,1); break;
             case 2: return _mm256_extract_epi32(p_vec,2); break;
             case 3: return _mm256_extract_epi32(p_vec,3); break;
             case 4: return _mm256_extract_epi32(p_vec,4); break;
             case 5: return _mm256_extract_epi32(p_vec,5); break;
             case 6: return _mm256_extract_epi32(p_vec,6); break;
             case 7: return _mm256_extract_epi32(p_vec,7); break;
         }
         return (typename avx2< v256< U > >::base_t)0;
      }
   };
}

#endif /* EXTRACT_AVX2_H */

