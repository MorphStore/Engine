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

#ifndef CALC_AVX512_H
#define CALC_AVX512_H



#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx512/extension_avx512.h>
#include <vector/primitives/calc.h>

#include <functional>

namespace vector{
    template<typename T>
    struct calc<avx512<v512<T>>, 64> {
     
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        add( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Add 64 bit integer values from two registers (avx512)" );
            return _mm512_add_epi64( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        sub( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Subtract 64 bit integer values (avx512)" );
            return _mm512_sub_epi64( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        add( avx512< v512< double > >::vector_t p_vec1,  avx512< v512< double > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Add double values from two registers (avx512)" );
            return _mm512_add_pd( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        sub( avx512< v512< double > >::vector_t p_vec1,  avx512< v512< double > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Subtract double values (avx512)" );
            return _mm512_sub_pd( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static uint64_t
        hadd( avx512< v512< uint64_t > >::vector_t p_vec1) {
            trace( "[VECTOR] - Subtract double values (avx512)" );
            return _mm512_reduce_add_epi64( p_vec1);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static double
        hadd( avx512< v512< double > >::vector_t p_vec1) {
            trace( "[VECTOR] - Subtract double values (avx512)" );
            return _mm512_reduce_add_pd( p_vec1);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        mul( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - multiply integer values (avx512)" );
            return _mm512_mul_epi32( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        div( avx512< v512< double > >::vector_t p_vec1,  avx512< v512< double > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - divide double values (avx512)" );
            return _mm512_div_pd( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        div( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            
            trace( "[VECTOR] - divide integer values (avx2)" );
            __m512d intermediate;
            __m512d divhelper=_mm512_set1_pd(0x0010000000000000);
            
            //Load as double and divide -> 64-bit integer division is not supported in sse or avx(2) 
            intermediate=_mm512_div_pd((__m512d)p_vec1,(__m512d)p_vec2);

            //Make an integer out of the double by adding a bit at the 52nd position and XORing the result bitwise with the bit at the 52nd position (all other bits are 0)
            intermediate=_mm512_add_pd(intermediate,divhelper);

            //return the result
            return _mm512_xor_si512(_mm512_castpd_si512(intermediate),_mm512_castpd_si512(divhelper));

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        mod( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            
             
                __m512d divhelper=_mm512_set1_pd(0x0010000000000000);
           
                //Cast to double
             
                __m512d left=(__m512d)p_vec1;
           
                __m512d right=(__m512d)p_vec2;
               
                //Divide -> 64-bit integer division is not supported in sse or avx 
                __m512d intermediate=_mm512_div_pd(left,right);
                
                //Floor
                intermediate=_mm512_floor_pd(intermediate);
                
                //Make an integer out of the double by adding a bit at the 52nd position and masking out mantisse and sign
                intermediate=_mm512_add_pd(intermediate,divhelper);
                __m512i mod_intermediate=_mm512_xor_si512(_mm512_castpd_si512(intermediate),_mm512_castpd_si512(divhelper));
               
                //multiply again
                mod_intermediate=_mm512_mul_epi32(mod_intermediate,p_vec2);
                
                //difference beween result and original
                return _mm512_sub_epi64(p_vec1,mod_intermediate);
                             
            }
      
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        inv( avx512< v512< int64_t > >::vector_t p_vec ) {
                
                return _mm512_sub_epi64(_mm512_set1_epi64(0),p_vec);
                             
        }
        
        template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        inv( avx512< v512< double > >::vector_t p_vec ) {
                
                return _mm512_sub_pd(_mm512_set1_pd(0),p_vec);
                             
        }
      
    };
    
    template<typename T>
    struct calc<avx512<v512<T>>, 32> {
     
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        add( avx512< v512< uint32_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Add 32 bit integer values from two registers (avx512)" );
            return _mm512_add_epi32( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        sub( avx512< v512< uint32_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Subtract 32 bit integer values (avx512)" );
            return _mm512_sub_epi32( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        add( avx512< v512< float > >::vector_t p_vec1,  avx512< v512< float > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Add float values (avx512)" );
            return _mm512_add_ps( p_vec1, p_vec2);

        }
        
         template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        sub( avx512< v512< float > >::vector_t p_vec1,  avx512< v512< float > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Subtract float values (avx512)" );
            return _mm512_sub_ps( p_vec1, p_vec2);
        }
         
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static uint64_t
        hadd( avx512< v512< uint32_t > >::vector_t p_vec1) {
            trace( "[VECTOR] - Subtract double values (avx512)" );
            return _mm512_reduce_add_epi32( p_vec1);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static float
        hadd( avx512< v512< float > >::vector_t p_vec1) {
            trace( "[VECTOR] - Subtract double values (avx512)" );
            return _mm512_reduce_add_ps( p_vec1);

        }
                 
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        mul( avx512< v512< uint32_t > >::vector_t p_vec1,  avx512< v512< uint32_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - multiply integer values (avx512)" );
            return _mm512_mullo_epi32( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        div( avx512< v512< float > >::vector_t p_vec1,  avx512< v512< float > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - divide float values (avx512)" );
            return _mm512_div_ps( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        inv( avx512< v512< int64_t > >::vector_t p_vec ) {
                
                return _mm512_sub_epi32(_mm512_set1_epi32(0),p_vec);
                             
        }
        
        template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx512< v512< U > >::vector_t
        inv( avx512< v512< float > >::vector_t p_vec ) {
                
                return _mm512_sub_ps(_mm512_set1_ps(0),p_vec);
                             
        }
    };
}
#endif /* CALC_AVX512_H */

