/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   manipulate_avx512.h
 * Author: Annett
 *
 * Created on 24. April 2019, 17:17
 */

#ifndef MORPHSTORE_VECTOR_SIMD_AVX512_PRIMITIVES_MANIPULATE_AVX512_H
#define MORPHSTORE_VECTOR_SIMD_AVX512_PRIMITIVES_MANIPULATE_AVX512_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include "../extension_avx512.h"
#include "../../../primitives/manipulate.h"

#include <functional>

namespace vectorlib{
    template<typename T>
    struct manipulate<avx512<v512<T>>, 64> {
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename avx512< v512< U > >::vector_t
        rotate( avx512< v512< uint64_t > >::vector_t p_vec ) {
            trace( "[VECTOR] - Rotate vector (sse)" );
             
            return _mm512_permutexvar_epi64(_mm512_set_epi64(6,5,4,3,2,1,0,7),p_vec);

        }
    };
}

#endif /* MORPHSTORE_VECTOR_SIMD_AVX512_PRIMITIVES_MANIPULATE_AVX512_H */

