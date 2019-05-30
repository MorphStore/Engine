/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   manipulate_scalar.h
 * Author: Annett
 *
 * Created on 27. Mai 2019, 17:24
 */

#ifndef MANIPULATE_SCALAR_H
#define MANIPULATE_SCALAR_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/primitives/manipulate.h>

#include <functional>

namespace vector{
    template<typename T>
    struct manipulate<scalar<v64<T>>, 64> {
        
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_INLINE
        static typename scalar< v64< U > >::vector_t
        rotate( scalar< v64< uint64_t > >::vector_t p_vec ) {
            trace( "[VECTOR] - Rotate vector (scalar)" );
             
            return p_vec;

        }
    };
}


#endif /* MANIPULATE_SCALAR_H */

