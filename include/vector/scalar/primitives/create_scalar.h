/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   create_scalar.h
 * Author: Annett
 *
 * Created on 21. Mai 2019, 13:17
 */

#ifndef CREATE_SCALAR_H
#define CREATE_SCALAR_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/primitives/create.h>

#include <functional>

namespace vectorlib {


   template<typename T>
   struct create<scalar<v64<T>>/*,64*/> {


      //template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64< T > >::vector_t
      set1( T a0) {
         trace( "[VECTOR] - set1 sse register." );
         return (typename scalar< v64< T > >::vector_t) (a0);
      }

      //template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar< v64< T > >::vector_t
      set_sequence( T a, MSV_CXX_ATTRIBUTE_PPUNUSED int b) {
         trace( "[VECTOR] - set_sequence sse register." );
         return a;
      }


   };


}


#endif /* CREATE_SCALAR_H */
