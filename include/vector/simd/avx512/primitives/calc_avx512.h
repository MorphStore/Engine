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
                 
    };
}
#endif /* CALC_AVX512_H */

