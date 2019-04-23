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
    template<typename T>
    struct compare<avx512<v512<T>>, 64> {
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static int
        equality( avx512< v512< uint64_t > >::vector_t p_vec1,  avx512< v512< uint64_t > >::vector_t p_vec2 ) {
            trace( "[VECTOR] - Compare 64 bit integer values from two registers (avx512)" );
             
            return _mm512_cmpeq_epi64_mask(p_vec1,p_vec2);

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
    };
}

#endif /* COMPARE_AVX512_H */

