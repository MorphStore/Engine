//
// Created by jpietrzyk on 10.05.19.
//

#ifndef MORPHSTORE_VECTOR_COMPLEX_HASH_H
#define MORPHSTORE_VECTOR_COMPLEX_HASH_H

#include <vector/general_vector.h>
#include <core/utils/preprocessor.h>
#include <vector/primitives/calc.h>
#include <vector/primitives/create.h>
namespace vector {

   template< class VectorExtension >
   struct multiply_mod_hash {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         vector_t const m_Prime;
         vector_t const m_Normalize;
         void set_normalize_value( base_t const p_Modulus ) {
            this->m_Normalize = set1<VectorExtension, vector_base_t_granularity::value>(p_Modulus);
         }
         state_t( base_t const p_Prime, base_t const p_Normalize ):
            m_Prime{set1<VectorExtension, vector_base_t_granularity::value>( p_Prime ) },
            m_Normalize{set1<VectorExtension, vector_base_t_granularity::value>( p_Normalize ) } {
         }
         state_t( base_t const p_Normalize ):
            m_Prime{set1<VectorExtension, vector_base_t_granularity::value>( ( 1 << 16 ) + 1 ) },
            m_Normalize{set1<VectorExtension, vector_base_t_granularity::value>( p_Normalize ) } {
         }
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_t apply_with_normalize(
         vector_t const & p_Key,
         typename multiply_mod_hash<VectorExtension>::state_t const & p_State
      ) {
         return
            mod<VectorExtension>::apply(
               mul<VectorExtension>::apply(
                  p_Key,
                  p_State.m_Prime
               ),
               p_State.m_Modulus
            );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_t apply(
         vector_t const & p_Key,
         typename multiply_mod_hash<VectorExtension>::state_t const & p_State
      ) {
         return
            mul<VectorExtension>::apply(
               p_Key,
               p_State.m_Prime
            );
      }

   };

   /*template<class VectorExtension>
   struct multiply_mod_hash {
      vector_t const m_Prime;
      vector_t const m_Mod;

      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      vector_t operator()( vector_t const & p_Key ) {
         return
            mod<VectorExtension>::apply(
               mul<VectorExtension>::apply(
                  p_Key,
                  m_Prime
               ),
               m_Mod
            );
      }
      multiply_hash( base_t p_Prime, base_t p_Mod ) :
         m_Prime{ set1<VectorExtension, vector_base_t_granularity::value>( p_Prime ) },
         m_Mod{ set1<VectorExtension, vector_base_t_granularity::value>( p_Mod ) } {

      }
   };*/


}
#endif //MORPHSTORE_VECTOR_COMPLEX_HASH_H
