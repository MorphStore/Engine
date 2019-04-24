/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   compare_avx2.h
 * Author: Annett
 *
 * Created on 23. April 2019, 16:57
 */

#ifndef COMPARE_AVX2_H
#define COMPARE_AVX2_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/primitives/compare.h>

#include <functional>

namespace vector{
    template<typename T>
    struct compare<avx2<v256<T>>, 64> {
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        equality( avx2< v256< uint64_t > >::vector_t p_vec1,  avx2< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (avx2)" );
             
            return _mm256_movemask_pd((__m256d)_mm256_cmpeq_epi64(p_vec1,p_vec2));

        }
    };
    
    template<typename T>
    struct compare<avx2<v256<T>>, 32> {
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        equality( avx2< v256< uint64_t > >::vector_t p_vec1,  avx2< v256< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 32 bit integer values from two registers (avx2)" );
             
            return _mm256_movemask_ps((__m256)_mm256_cmpeq_epi32(p_vec1,p_vec2));

        }
    };
}

#endif /* COMPARE_AVX2_H */

