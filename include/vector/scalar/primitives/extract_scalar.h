/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   extract_scalar.h
 * Author: Annett
 *
 * Created on 28. Mai 2019, 16:01
 */

#ifndef EXTRACT_SCALAR_H
#define EXTRACT_SCALAR_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/primitives/extract.h>

#include <functional>

namespace vector {
    
    
   template<typename T>
   struct extract<scalar<v64<T>>,64> {
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar< v64< U > >::base_t
      extract_value( scalar< v64< uint64_t > >::vector_t p_vec, int idx) {
         trace( "[VECTOR] - extract value from scalar register." );
         return p_vec;
      }
   };
   

}

#endif /* EXTRACT_SCALAR_H */

