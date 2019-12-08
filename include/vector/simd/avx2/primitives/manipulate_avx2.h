/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   manipulate_avx2.h
 * Author: Annett
 *
 * Created on 24. April 2019, 17:17
 */

#ifndef MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_MANIPULATE_AVX2_H
#define MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_MANIPULATE_AVX2_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/primitives/manipulate.h>

#include <functional>

namespace vectorlib{
    template<typename T>
    struct manipulate<avx2<v256<T>>, 64> {
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename avx2< v256< U > >::vector_t
        rotate( avx2< v256< uint64_t > >::vector_t p_vec ) {
            trace( "[VECTOR] - Rotate vector (avx2)" );
             
            return _mm256_permute4x64_epi64(p_vec,57); 

        }
    };

    template<typename T>
    struct manipulate<avx2<v256<T>>, 32> {
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename avx2< v256< U > >::vector_t
        rotate( avx2< v256< uint32_t > >::vector_t p_vec ) {
            trace( "[VECTOR] - Rotate vector (avx2)" );
             
            return _mm256_permutevar8x32_epi32(p_vec,_mm256_set_epi32(0,7,6,5,4,3,2,1)); 
        }
    };
}

#endif /* MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_MANIPULATE_AVX2_H */

