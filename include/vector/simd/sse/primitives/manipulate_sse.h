/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   manipulate_sse.h
 * Author: Annett
 *
 * Created on 24. April 2019, 17:17
 */

#ifndef MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_MANIPULATE_SSE_H
#define MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_MANIPULATE_SSE_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/sse/extension_sse.h>
#include <vector/primitives/manipulate.h>

#include <functional>

namespace vectorlib{
    template<typename T>
    struct manipulate<sse<v128<T>>, 64> {
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename sse< v128< U > >::vector_t
        rotate( sse< v128< uint64_t > >::vector_t p_vec ) {
            trace( "[VECTOR] - Rotate vector (sse)" );
             
            return (__m128i)(_mm_permute_pd((__m128d)p_vec,1));

        }
    };
}

#endif /* MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_MANIPULATE_SSE_H */

