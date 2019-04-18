/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   calc_sse.h
 * Author: Annett
 *
 * Created on 17. April 2019, 11:07
 */

#ifndef CALC_SSE_H
#define CALC_SSE_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/sse/extension_sse.h>
#include <vector/primitives/calc.h>

#include <functional>

namespace vector{
    template<typename T>
    struct calc<sse<v128<T>>, 64> {
     
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename sse< v128< U > >::vector_t
        add( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Add 64 bit integer values from two registers (sse)" );
            return _mm_add_epi64( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename sse< v128< U > >::vector_t
        sub( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Subtract 64 bit integer values (sse)" );
            return _mm_sub_epi64( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename sse< v128< U > >::vector_t
        add( sse< v128< double > >::vector_t p_vec1,  sse< v128< double > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Add double values from two registers (sse)" );
            return _mm_add_pd( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename sse< v128< U > >::vector_t
        sub( sse< v128< double > >::vector_t p_vec1,  sse< v128< double > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Subtract double values (sse)" );
            return _mm_sub_pd( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static uint64_t
        hadd( sse< v128< uint64_t > >::vector_t p_vec1) {
            trace( "[VECTOR] - Subtract double values (avx512)" );
                
            return _mm_extract_epi64((__m128i)_mm_hadd_pd((__m128d)p_vec1,(__m128d)p_vec1),0);

        }
        
                
    };
    
    template<typename T>
    struct calc<sse<v128<T>>, 32> {
     
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename sse< v128< U > >::vector_t
        add( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Add 32 bit integer values from two registers (sse)" );
            return _mm_add_epi32( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename sse< v128< U > >::vector_t
        sub( sse< v128< uint64_t > >::vector_t p_vec1,  sse< v128< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Subtract 32 bit integer values (sse)" );
            return _mm_sub_epi32( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename sse< v128< U > >::vector_t
        add( sse< v128< float > >::vector_t p_vec1,  sse< v128< float > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Add float values (sse)" );
            return _mm_add_ps( p_vec1, p_vec2);

        }
        
        template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename sse< v128< U > >::vector_t
        sub( sse< v128< float > >::vector_t p_vec1,  sse< v128< float > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Subtract float values (sse)" );
            return _mm_sub_ps( p_vec1, p_vec2);
        }
         
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static uint64_t
        hadd( sse< v128< uint32_t > >::vector_t p_vec1) {
            trace( "[VECTOR] - Subtract double values (avx512)" );
            __m128i a=_mm_hadd_epi32(p_vec1,p_vec1);    
            return _mm_extract_epi32(a,0)+_mm_extract_epi32(a,1);

        }
        
                template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0  >
        MSV_CXX_ATTRIBUTE_INLINE
        static float
        hadd( sse< v128< float > >::vector_t p_vec1) {
            trace( "[VECTOR] - Subtract double values (avx512)" );
            __m128 a=_mm_hadd_ps(p_vec1,p_vec1);    
            return _mm_extract_ps(a,0)+_mm_extract_epi32(a,1);

        }
    };
}
#endif /* CALC_SSE_H */

