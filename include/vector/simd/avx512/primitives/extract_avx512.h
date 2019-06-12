/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   extract_avx512.h
 * Author: Annett
 *
 * Created on 25. April 2019, 14:00
 */

#ifndef EXTRACT_AVX512_H
#define EXTRACT_AVX512_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx512/extension_avx512.h>
#include <vector/primitives/extract.h>

#include <functional>

namespace vector {
    
    
   template<typename T>
   struct extract<avx512<v512<T>>,64> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx512< v512< U > >::base_t
      extract_value( avx512< v512< uint64_t > >::vector_t p_vec, int idx) {
         trace( "[VECTOR] - extract value from sse register." );
         switch (idx){
             case 0: return _mm256_extract_epi64(_mm512_extracti64x4_epi64(p_vec,0),0); break;
             case 1: return _mm256_extract_epi64(_mm512_extracti64x4_epi64(p_vec,0),1); break;
             case 2: return _mm256_extract_epi64(_mm512_extracti64x4_epi64(p_vec,0),2); break;
             case 3: return _mm256_extract_epi64(_mm512_extracti64x4_epi64(p_vec,0),3); break;
             case 4: return _mm256_extract_epi64(_mm512_extracti64x4_epi64(p_vec,1),0); break;
             case 5: return _mm256_extract_epi64(_mm512_extracti64x4_epi64(p_vec,1),1); break;
             case 6: return _mm256_extract_epi64(_mm512_extracti64x4_epi64(p_vec,1),2); break;
             case 7: return _mm256_extract_epi64(_mm512_extracti64x4_epi64(p_vec,1),3); break;
         }
         return _mm256_extract_epi64(_mm512_extracti64x4_epi64(p_vec,0),0);
      }
   };
   
   template<typename T>
   struct extract<avx512<v512<T>>,32> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx512< v512< U > >::base_t
      extract_value( avx512< v512< uint64_t > >::vector_t p_vec, int idx) {
         trace( "[VECTOR] - extract value from sse register." );
         switch (idx){
             case 0: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,0),0); break;
             case 1: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,0),1); break;
             case 2: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,0),2); break;
             case 3: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,0),3); break;
             case 4: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,0),4); break;
             case 5: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,0),5); break;
             case 6: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,0),6); break;
             case 7: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,0),7); break;
             case 8: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,1),0); break;
             case 9: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,1),1); break;
             case 10: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,1),2); break;
             case 11: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,1),3); break;
             case 12: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,1),4); break;
             case 13: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,1),5); break;
             case 14: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,1),6); break;
             case 15: return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,1),7); break;
         }
         return _mm256_extract_epi32(_mm512_extracti64x4_epi64(p_vec,0),0);
     }
   };
}

#endif /* EXTRACT_AVX512_H */

