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

#ifndef CALC_AVX2_H
#define CALC_AVX2_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/primitives/calc.h>

#include <functional>

namespace vector{
    template<typename T>
    struct calc<avx2<v256<T>>, 64> {
     
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        add( avx2< v256< uint64_t > >::vector_t p_vec1,  avx2< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Add 64 bit integer values from two registers (avx2)" );
            return _mm256_add_epi64( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        sub( avx2< v256< uint64_t > >::vector_t p_vec1,  avx2< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Subtract 64 bit integer values (avx2)" );
            return _mm256_sub_epi64( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        add( avx2< v256< double > >::vector_t p_vec1,  avx2< v256< double > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Add double values from two registers (avx2)" );
            return _mm256_add_pd( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        sub( avx2< v256< double > >::vector_t p_vec1,  avx2< v256< double > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Subtract double values (avx2)" );
            return _mm256_sub_pd( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static uint64_t
        hadd( avx2< v256< uint32_t > >::vector_t p_vec1) {
            trace( "[VECTOR] - Subtract double values (avx512)" );
            __m256i a= (__m256i)_mm256_hadd_pd((__m256d)p_vec1,(__m256d)p_vec1);            
            return _mm256_extract_epi64(a,0)+_mm256_extract_epi64(a,2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        mul( avx2< v256< uint64_t > >::vector_t p_vec1,  avx2< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - multiply integer values (avx2)" );
            return _mm256_mul_epi32( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        div( avx2< v256< double > >::vector_t p_vec1,  avx2< v256< double > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - divide float values (avx2)" );
            return _mm256_div_pd( p_vec1, p_vec2);

        }
        
                
        
        
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        div( avx2< v256< uint64_t > >::vector_t p_vec1,  avx2< v256< uint64_t > >::vector_t p_vec2 ) {
            
            trace( "[VECTOR] - divide integer values (avx2)" );
            __m256d intermediate;
            __m256d divhelper=_mm256_set1_pd(0x0010000000000000);
            
            //Load as double and divide -> 64-bit integer division is not supported in sse or avx 
            intermediate=_mm256_div_pd((__m256d)p_vec1,(__m256d)p_vec2);

            //Make an integer out of the double by adding a bit at the 52nd position and XORing the result bitwise with the bit at the 52nd position (all other bits are 0)
            intermediate=_mm256_add_pd(intermediate,divhelper);

            //return the result
            return _mm256_xor_si256(_mm256_castpd_si256(intermediate),_mm256_castpd_si256(divhelper));

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        mod( avx2< v256< uint64_t > >::vector_t p_vec1,  avx2< v256< uint64_t > >::vector_t p_vec2 ) {
            
             
                __m256d divhelper=_mm256_set1_pd(0x0010000000000000);
           
                //Cast to double
             
                __m256d left=(__m256d)p_vec1;
           
                __m256d right=(__m256d)p_vec2;
               
                //Divide -> 64-bit integer division is not supported in sse or avx 
                __m256d intermediate=_mm256_div_pd(left,right);
                
                //Floor
                intermediate=_mm256_floor_pd(intermediate);
                
                //Make an integer out of the double by adding a bit at the 52nd position and masking out mantisse and sign
                intermediate=_mm256_add_pd(intermediate,divhelper);
                __m256i mod_intermediate=_mm256_xor_si256(_mm256_castpd_si256(intermediate),_mm256_castpd_si256(divhelper));
               
                //multiply again
                mod_intermediate=_mm256_mul_epi32(mod_intermediate,p_vec2);
                
                //difference beween result and original
                return _mm256_sub_epi64(p_vec1,mod_intermediate);
                             
            }
    };
    
    template<typename T>
    struct calc<avx2<v256<T>>, 32> {
     
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        add( avx2< v256< uint64_t > >::vector_t p_vec1,  avx2< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Add 32 bit integer values from two registers (avx2)" );
            return _mm256_add_epi32( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        sub( avx2< v256< uint64_t > >::vector_t p_vec1,  avx2< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Subtract 32 bit integer values (avx2)" );
            return _mm256_sub_epi32( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        add( avx2< v256< float > >::vector_t p_vec1,  avx2< v256< float > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Add float values (avx2)" );
            return _mm256_add_ps( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        sub( avx2< v256< float > >::vector_t p_vec1,  avx2< v256< float > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Subtract float values (avx2)" );
            return _mm256_sub_ps( p_vec1, p_vec2);
        }
         
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static uint64_t
        hadd( avx2< v256< uint32_t > >::vector_t p_vec1) {
            trace( "[VECTOR] - Subtract double values (avx512)" );
            __m256i a=_mm256_hadd_epi32(p_vec1,p_vec1);            
            return _mm256_extract_epi32(a,0)+_mm256_extract_epi32(a,1)+_mm256_extract_epi32(a,4)+_mm256_extract_epi32(a,5);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        mul( avx2< v256< uint32_t > >::vector_t p_vec1,  avx2< v256< uint32_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - multiply integer values (avx2)" );
            return _mm256_mullo_epi32( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename avx2< v256< U > >::vector_t
        div( avx2< v256< float > >::vector_t p_vec1,  avx2< v256< float > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - divide float values (avx2)" );
            return _mm256_div_ps( p_vec1, p_vec2);

        }
        
        

    };
}

#endif /* CALC_AVX2_H */

