/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   compare_avx512.h
 * Author: Annett
 *
 * Created on 23. April 2019, 16:57
 */

#ifndef COMPARE_AVX512_H
#define COMPARE_AVX512_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx512/extension_avx512.h>
#include <vector/primitives/compare.h>

#include <functional>

namespace vector{
   template<>
   struct equal<avx512<v512<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512<v512<uint64_t>>::mask_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: == ? (avx512)" );
         return _mm512_cmpeq_epi64_mask(p_vec1, p_vec2);
      }
   };
   template<>
   struct less<avx512<v512<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512<v512<uint64_t>>::mask_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: < ? (avx512)" );
         return _mm512_cmplt_epi64_mask(p_vec1, p_vec2);
      }
   };
   template<>
   struct lessequal<avx512<v512<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512<v512<uint64_t>>::mask_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: <= ? (avx512)" );
         return _mm512_cmple_epi64_mask(p_vec1, p_vec2);
      }
   };

   template<>
   struct greater<avx512<v512<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512<v512<uint64_t>>::mask_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: > ? (avx512)" );
         return _mm512_cmpgt_epi64_mask(p_vec1, p_vec2);
      }
   };
   template<>
   struct greaterequal<avx512<v512<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx512<v512<uint64_t>>::mask_t
      apply(
         typename avx512<v512<uint64_t>>::vector_t const & p_vec1,
         typename avx512<v512<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: >= ? (avx512)" );
         return _mm512_cmpge_epi64_mask(p_vec1, p_vec2);
      }
   };
   template<>
   struct count_matches<avx512<v512<uint64_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static uint8_t
      apply(
         typename avx512<v512<uint64_t>>::mask_t const & p_mask
      ) {
         trace( "[VECTOR] - Count matches in a comparison mask (avx512)" );
         // @todo Which one is faster?
         // return __builtin_popcount(p_mask);
         return _mm_popcnt_u32(p_mask);
      }
   };
   
   /*
    template<typename T>
    struct compare<avx512<v512<T>>, 64> {
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        equality( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (avx512)" );
             
            return _mm512_cmpeq_epi64_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessthan( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm512_cmplt_epi64_mask(p_vec1,p_vec2);

        }
                
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterthan( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm512_cmpgt_epi64_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterequal( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm512_cmpge_epi64_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessequal( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm512_cmple_epi64_mask(p_vec1,p_vec2);

        }
                
        
    };
    
    template<typename T>
    struct compare<avx512<v512<T>>, 32> {
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        equality( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (avx512)" );
             
            return _mm512_cmpeq_epi32_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessthan( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm512_cmplt_epi32_mask(p_vec1,p_vec2);

        }
                
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterthan( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm512_cmpgt_epi32_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterequal( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm512_cmpge_epi32_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessequal( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm512_cmple_epi32_mask(p_vec1,p_vec2);

        }
    };
    
    template<typename T>
    struct compare<avx512<v256<T>>, 64> {
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        equality( avx512< v256< uint64_t > >::vector_t p_vec1,  avx512< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (avx512)" );
             
            return _mm256_cmpeq_epi64_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessthan( avx512< v256< uint64_t > >::vector_t p_vec1,  avx512< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm256_cmplt_epi64_mask(p_vec1,p_vec2);

        }
                
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterthan( avx512< v256< uint64_t > >::vector_t p_vec1,  avx512< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm256_cmpgt_epi64_mask(p_vec1,p_vec2);

        } 
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterequal( avx512< v256< uint64_t > >::vector_t p_vec1,  avx512< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm256_cmpge_epi64_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessequal( avx512< v256< uint64_t > >::vector_t p_vec1,  avx512< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm256_cmple_epi64_mask(p_vec1,p_vec2);

        }
    };
     
    template<typename T>
    struct compare<avx512<v128<T>>, 64> {
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        equality( avx512< v128< uint64_t > >::vector_t p_vec1,  avx512< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (avx512)" );
             
            return _mm_cmpeq_epi64_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessthan( avx512< v128< uint64_t > >::vector_t p_vec1,  avx512< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm_cmplt_epi64_mask(p_vec1,p_vec2);

        }
                
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterthan( avx512< v128< uint64_t > >::vector_t p_vec1,  avx512< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm_cmpgt_epi64_mask(p_vec1,p_vec2);

        } 
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterequal( avx512< v128< uint64_t > >::vector_t p_vec1,  avx512< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm_cmpge_epi64_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessequal( avx512< v128< uint64_t > >::vector_t p_vec1,  avx512< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm_cmple_epi64_mask(p_vec1,p_vec2);

        }
    };
    
    
    template<typename T>
    struct compare<avx512<v256<T>>, 32> {
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        equality( avx512< v256< uint64_t > >::vector_t p_vec1,  avx512< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (avx512)" );
             
            return _mm256_cmpeq_epi32_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessthan( avx512< v256< uint64_t > >::vector_t p_vec1,  avx512< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm256_cmplt_epi32_mask(p_vec1,p_vec2);

        }
                
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterthan( avx512< v256< uint64_t > >::vector_t p_vec1,  avx512< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm256_cmpgt_epi32_mask(p_vec1,p_vec2);

        } 
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterequal( avx512< v256< uint64_t > >::vector_t p_vec1,  avx512< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm256_cmpge_epi32_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessequal( avx512< v256< uint64_t > >::vector_t p_vec1,  avx512< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm256_cmple_epi32_mask(p_vec1,p_vec2);

        }
    };
     
    template<typename T>
    struct compare<avx512<v128<T>>, 32> {
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        equality( avx512< v128< uint64_t > >::vector_t p_vec1,  avx512< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (avx512)" );
             
            return _mm_cmpeq_epi32_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessthan( avx512< v128< uint64_t > >::vector_t p_vec1,  avx512< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm_cmplt_epi32_mask(p_vec1,p_vec2);

        }
                
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterthan( avx512< v128< uint64_t > >::vector_t p_vec1,  avx512< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm_cmpgt_epi32_mask(p_vec1,p_vec2);

        } 
                
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterequal( avx512< v128< uint64_t > >::vector_t p_vec1,  avx512< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm_cmpge_epi32_mask(p_vec1,p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessequal( avx512< v128< uint64_t > >::vector_t p_vec1,  avx512< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm_cmple_epi32_mask(p_vec1,p_vec2);

        }
    };*/
}

#endif /* COMPARE_AVX512_H */

