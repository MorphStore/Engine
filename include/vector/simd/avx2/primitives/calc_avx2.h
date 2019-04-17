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
    };
}

#endif /* CALC_AVX2_H */

