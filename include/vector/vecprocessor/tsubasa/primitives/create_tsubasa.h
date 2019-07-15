//
// Created by jpietrzyk on 15.07.19.
//

#ifndef MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_CREATE_TSUBASA_H
#define MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_CREATE_TSUBASA_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <core/utils/variadic.h>
#include <vector/vecprocessor/tsubasa/extension_tsubasa.h>
#include <vector/primitives/create.h>

#include <functional>

namespace vector {

   template< typename T >
   struct create< aurora< v16k< T > >, 64 > {

      template<
         typename U = T,
         typename std::enable_if<
            ( std::is_integral< U >::value ) &&
            ( sizeof( U ) == 8 ),
            int
         >::type = 0
      >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename aurora< v16k< T > >::vector_t
      set1( T a ) {
         _ve_vbrd_vs_i64( a );
      }

      template<
         typename U = T,
         typename std::enable_if<
            ( std::is_integral< U >::value ) &&
            ( sizeof( U ) == 4 ),
            int
         >::type = 0
      >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename aurora< v16k< T > >::vector_t
      set1( T a ) {
         _ve_vbrdl_vs_i32( a );
      }

      template<
         typename U = T,
         typename std::enable_if<
            ( typename std::enable_if< std::is_integral< U >::value ) &&
            ( sizeof( U ) == 8 ),
            int
         >::type = 0
      >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename aurora< v16k< T > >::vector_t
      set_sequence( U const p_Start, U const p_StepWidth) {
         trace( "[VECTOR] - set_sequence tsubasa register." );
         return _ve_vld_vss(
            sizeof( T ),
            (
               morphstore::static_sequence_array< 8, U, 256 >( p_Start, p_StepWidth )
            ).data()
         );
      }

   };
}

#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_CREATE_TSUBASA_H