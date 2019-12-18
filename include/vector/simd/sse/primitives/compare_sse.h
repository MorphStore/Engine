/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   compare_sse.h
 * Author: Annett
 *
 * Created on 23. April 2019, 16:56
 */

#ifndef MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_COMPARE_SSE_H
#define MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_COMPARE_SSE_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/sse/extension_sse.h>
#include <vector/primitives/compare.h>

#include <functional>

namespace vectorlib{
   template<>
   struct equal<sse<v128<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint64_t>>::mask_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const p_vec1,
         typename sse<v128<uint64_t>>::vector_t const p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: == ? (sse)" );
         return
            _mm_movemask_pd(
               _mm_castsi128_pd(
                  _mm_cmpeq_epi64(p_vec1, p_vec2)
               )
            );
      }
   };
   template<>
   struct less<sse<v128<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint64_t>>::mask_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1,
         typename sse<v128<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: < ? (sse)" );
         return
            _mm_movemask_pd(
               _mm_castsi128_pd(
                  _mm_cmpgt_epi64(p_vec2, p_vec1)
               )
            );
      }
   };
   template<>
   struct lessequal<sse<v128<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint64_t>>::mask_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1,
         typename sse<v128<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: <= ? (sse)" );
         return
            _mm_movemask_pd(
               _mm_castsi128_pd(
                  _mm_or_si128(
                     _mm_cmpeq_epi64(p_vec1, p_vec2),
                     _mm_cmpgt_epi64(p_vec2, p_vec1)
                  )
               )
            );
      }
   };

   template<>
   struct greater<sse<v128<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint64_t>>::mask_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1,
         typename sse<v128<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: > ? (sse)" );
         return
            _mm_movemask_pd(
               _mm_castsi128_pd(
                  _mm_cmpgt_epi64(p_vec1, p_vec2)
               )
            );
      }
   };
   template<>
   struct greaterequal<sse<v128<uint64_t>>, 64> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint64_t>>::mask_t
      apply(
         typename sse<v128<uint64_t>>::vector_t const & p_vec1,
         typename sse<v128<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: >= ? (sse)" );
         return
            _mm_movemask_pd(
               _mm_castsi128_pd(
                  _mm_or_si128(
                     _mm_cmpeq_epi64(p_vec1, p_vec2),
                     _mm_cmpgt_epi64(p_vec1, p_vec2)
                  )
               )
            );
      }
   };
   template<>
   struct count_matches<sse<v128<uint64_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static uint8_t
      apply(
         typename sse<v128<uint64_t>>::mask_t const & p_mask
      ) {
         trace( "[VECTOR] - Count matches in a comparison mask (sse)" );
         // @todo Which one is faster?
         // return __builtin_popcount(p_mask);
         return _mm_popcnt_u64(p_mask);
      }
   };

   template<>
   struct equal<sse<v128<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint32_t>>::mask_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const p_vec1,
         typename sse<v128<uint32_t>>::vector_t const p_vec2
      ) {
         trace( "[VECTOR] - Compare 32 bit integer values from two registers: == ? (sse)" );
         return
            _mm_movemask_ps(
               _mm_castsi128_ps(
                  _mm_cmpeq_epi32(p_vec1, p_vec2)
               )
            );
      }
   };
   template<>
   struct less<sse<v128<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint32_t>>::mask_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_vec1,
         typename sse<v128<uint32_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 32 bit integer values from two registers: < ? (sse)" );
         return
            _mm_movemask_ps(
               _mm_castsi128_ps(
                  _mm_cmpgt_epi32(p_vec2, p_vec1)
               )
            );
      }
   };
   template<>
   struct lessequal<sse<v128<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint32_t>>::mask_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_vec1,
         typename sse<v128<uint32_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 32 bit integer values from two registers: <= ? (sse)" );
         return
            _mm_movemask_ps(
               _mm_castsi128_ps(
                  _mm_or_si128(
                     _mm_cmpeq_epi32(p_vec1, p_vec2),
                     _mm_cmpgt_epi32(p_vec2, p_vec1)
                  )
               )
            );
      }
   };

   template<>
   struct greater<sse<v128<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint32_t>>::mask_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_vec1,
         typename sse<v128<uint32_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 32 bit integer values from two registers: > ? (sse)" );
         return
            _mm_movemask_ps(
               _mm_castsi128_ps(
                  _mm_cmpgt_epi32(p_vec1, p_vec2)
               )
            );
      }
   };
   template<>
   struct greaterequal<sse<v128<uint32_t>>, 32> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint32_t>>::mask_t
      apply(
         typename sse<v128<uint32_t>>::vector_t const & p_vec1,
         typename sse<v128<uint32_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 32 bit integer values from two registers: >= ? (sse)" );
         return
            _mm_movemask_ps(
               _mm_castsi128_ps(
                  _mm_or_si128(
                     _mm_cmpeq_epi32(p_vec1, p_vec2),
                     _mm_cmpgt_epi32(p_vec1, p_vec2)
                  )
               )
            );
      }
   };
   template<>
   struct count_matches<sse<v128<uint32_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static uint8_t
      apply(
         typename sse<v128<uint32_t>>::mask_t const & p_mask
      ) {
         trace( "[VECTOR] - Count matches in a comparison mask (sse)" );
         // @todo Which one is faster?
         // return __builtin_popcount(p_mask);
         return _mm_popcnt_u64(p_mask);
      }
   };

//no 16 bit equivalents for _mm_movemask_pd

   template<>
   struct equal<sse<v128<uint8_t>>, 8> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint8_t>>::mask_t
      apply(
         typename sse<v128<uint8_t>>::vector_t const p_vec1,
         typename sse<v128<uint8_t>>::vector_t const p_vec2
      ) {
         trace( "[VECTOR] - Compare 8 bit integer values from two registers: == ? (sse)" );
         return
            _mm_movemask_epi8(
                  _mm_cmpeq_epi8(p_vec1, p_vec2)
               
            );
      }
   };
   template<>
   struct less<sse<v128<uint8_t>>, 8> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint8_t>>::mask_t
      apply(
         typename sse<v128<uint8_t>>::vector_t const & p_vec1,
         typename sse<v128<uint8_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 8 bit integer values from two registers: < ? (sse)" );
         return
            _mm_movemask_epi8(
                  _mm_cmpgt_epi8(p_vec2, p_vec1)               
            );
      }
   };
   template<>
   struct lessequal<sse<v128<uint8_t>>, 8> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint8_t>>::mask_t
      apply(
         typename sse<v128<uint8_t>>::vector_t const & p_vec1,
         typename sse<v128<uint8_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 8 bit integer values from two registers: <= ? (sse)" );
         return
            _mm_movemask_epi8(
                  _mm_or_si128(
                     _mm_cmpeq_epi8(p_vec1, p_vec2),
                     _mm_cmpgt_epi8(p_vec2, p_vec1)
               )
            );
      }
   };

   template<>
   struct greater<sse<v128<uint8_t>>, 8> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint8_t>>::mask_t
      apply(
         typename sse<v128<uint8_t>>::vector_t const & p_vec1,
         typename sse<v128<uint8_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 8 bit integer values from two registers: > ? (sse)" );
         return
            _mm_movemask_epi8(
                  _mm_cmpgt_epi8(p_vec1, p_vec2)
               
            );
      }
   };
   template<>
   struct greaterequal<sse<v128<uint8_t>>, 8> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse<v128<uint8_t>>::mask_t
      apply(
         typename sse<v128<uint8_t>>::vector_t const & p_vec1,
         typename sse<v128<uint8_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 8 bit integer values from two registers: >= ? (sse)" );
         return
            _mm_movemask_epi8(
                  _mm_or_si128(
                     _mm_cmpeq_epi8(p_vec1, p_vec2),
                     _mm_cmpgt_epi8(p_vec1, p_vec2)
                  
               )
            );
      }
   };
   template<>
   struct count_matches<sse<v128<uint8_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static uint8_t
      apply(
         typename sse<v128<uint8_t>>::mask_t const & p_mask
      ) {
         trace( "[VECTOR] - Count matches in a comparison mask (sse)" );
         // @todo Which one is faster?
         // return __builtin_popcount(p_mask);
         return _mm_popcnt_u64(p_mask);
      }
   };

   /*
    template<typename T>
    struct compare<sse<v128<T>>, 64> {
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        equality( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm_movemask_pd((__m128d)_mm_cmpeq_epi64(p_vec1,p_vec2));

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessthan( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm_movemask_pd((__m128d)_mm_cmpgt_epi64(p_vec2,p_vec1));

        }
                
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterthan( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm_movemask_pd((__m128d)_mm_cmpgt_epi64(p_vec1,p_vec2));

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterequal( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm_movemask_pd((__m128d)(_mm_or_si128(_mm_cmpeq_epi64(p_vec1,p_vec2),_mm_cmpgt_epi64(p_vec1,p_vec2))));

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessequal( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (sse)" );
             
            return _mm_movemask_pd((__m128d)(_mm_or_si128(_mm_cmpeq_epi64(p_vec1,p_vec2),_mm_cmpgt_epi64(p_vec2,p_vec1))));

        }
        
    };
    
    template<typename T>
    struct compare<sse<v128<T>>, 32> {
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        equality( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm_movemask_ps((__m128)_mm_cmpeq_epi32(p_vec1,p_vec2));

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessthan( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm_movemask_ps((__m128)_mm_cmpgt_epi64(p_vec2,p_vec1));

        }
                
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterthan( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm_movemask_ps((__m128)_mm_cmpgt_epi64(p_vec1,p_vec2));

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        greaterequal( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm_movemask_ps((__m128)(_mm_or_si128(_mm_cmpeq_epi32(p_vec1,p_vec2),_mm_cmpgt_epi32(p_vec1,p_vec2))));

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        lessequal( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (sse)" );
             
            return _mm_movemask_ps((__m128)(_mm_or_si128(_mm_cmpeq_epi32(p_vec1,p_vec2),_mm_cmpgt_epi32(p_vec2,p_vec1))));

        }
    };*/
}

#endif /* MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_COMPARE_SSE_H */

