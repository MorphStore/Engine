//
// Created by jpietrzyk on 15.07.19.
//

#ifndef MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_CALC_TSUBASA_H
#define MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_CALC_TSUBASA_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/vecprocessor/tsubasa/extension_tsubasa.h>
#include <vector/primitives/calc.h>

namespace vectorlib {

   template< typename T >
   struct add< aurora< v16k< T > >, 64 > {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename aurora< v16k< T > >::vector_t
      apply(
         typename aurora< v16k< T > >::vector_t const & p_Vec1,
         typename aurora< v16k< T > >::vector_t const & p_Vec2
      ) {
         trace( "[VECTOR] - Add 64 bit integer values from two registers (tsubasa)" );
         return _ve_vaddul_vvv( p_Vec1, p_Vec2 );
      }
   };
}



#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_CALC_TSUBASA_H
