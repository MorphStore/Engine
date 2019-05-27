//
// Created by jpietrzyk on 10.05.19.
//

#ifndef MORPHSTORE_VECTOR_COMPLEX_HASH_H
#define MORPHSTORE_VECTOR_COMPLEX_HASH_H

#include <vector/general_vector.h>
#include <core/utils/preprocessor.h>
#include <vector/primitives/calc.h>
#include <vector/primitives/create.h>
#include <vector/primitives/logic.h>
#include <core/utils/math.h>
namespace vector {



   template< class VectorExtension >
   struct multiply_mod_hash {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         vector_t const m_Prime;
         state_t( base_t const p_Prime = ( ( 1 << 16 ) + 1 ) ):
            m_Prime{set1<VectorExtension, vector_base_t_granularity::value>( p_Prime ) }{ }
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_t apply(
         vector_t const & p_Key,
         state_t const & p_State
      ) {
         return
            mul<VectorExtension>::apply(
               p_Key,
               p_State.m_Prime
            );
      }
   };

}
#endif //MORPHSTORE_VECTOR_COMPLEX_HASH_H
