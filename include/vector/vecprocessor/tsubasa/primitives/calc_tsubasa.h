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
//      static_assert( std::is_integral< T >::value, "Type has to be integral.");
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename aurora< v16k< T > >::vector_t
      apply(
         typename aurora< v16k< T > >::vector_t const & p_Vec1,
         typename aurora< v16k< T > >::vector_t const & p_Vec2
      ) {
         _ve_lvl(256);
         trace( "[VECTOR] - Add 64 bit integer values from two registers. (aurora tsubasa)" );
         return _ve_vaddul_vvv( p_Vec1, p_Vec2 );
      }
   };

   template< typename T >
   struct mul<aurora< v16k< T > >, 64> {
//      static_assert( std::is_integral< T >::value, "Type has to be integral.");
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename aurora< v16k< T > >::vector_t
      apply(
         typename aurora< v16k< T > >::vector_t const & p_Vec1,
         typename aurora< v16k< T > >::vector_t const & p_Vec2
      ){
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers. (aurora tsubasa)" );
         return _ve_vmulul_vvv( p_Vec1, p_Vec2 );
      }
   };
}



#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_CALC_TSUBASA_H
