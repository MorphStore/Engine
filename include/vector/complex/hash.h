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


   enum class size_policy_set {
      ARBITRARY,
      EXPONENTIAL
   };
   enum class growth_policy_set {
      NONE
   };

   template<
      size_policy_set SPS,
      growth_policy_set GPS,
      int TargetLoadFactor = 60
   >
   struct size_strategy_set {

   };






   template< class VectorExtension >
   struct multiply_mod_hash {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      template< class BiggestSupportedVectorExtension >
      struct state_t {
         vector_t const m_Prime;
         vector_t const m_Normalize;
         vector_t const m_Alignment;
         vector_t const m_AlignmentHelper;
         vector_t const m_AlignmentHelperComplement;
         vector_t const m_Zero;
         void set_normalize_value( base_t const p_Modulus ) {
            this->m_Normalize = set1<VectorExtension, vector_base_t_granularity::value>(p_Modulus);
         }
         state_t( base_t const p_Prime, base_t const p_Normalize ):
            m_Prime{set1<VectorExtension, vector_base_t_granularity::value>( p_Prime ) },
            m_Normalize{set1<VectorExtension, vector_base_t_granularity::value>( p_Normalize ) },
            m_Alignment{set1<VectorExtension, vector_base_t_granularity::value>(
               BiggestSupportedVectorExtension::vector_helper_t::element_count::value
            )},
            m_AlignmentHelper{set1<VectorExtension, vector_base_t_granularity::value>(
               BiggestSupportedVectorExtension::vector_helper_t::element_count::value - 1
            )},
            m_AlignmentHelperComplement{set1<VectorExtension, vector_base_t_granularity::value>(
               - BiggestSupportedVectorExtension::vector_helper_t::element_count::value
            )},
            m_Zero{set1<VectorExtension, vector_base_t_granularity::value>(
               0
            )}{ }
         state_t( base_t const p_Normalize ):
            m_Prime{set1<VectorExtension, vector_base_t_granularity::value>( ( 1 << 16 ) + 1 ) },
            m_Normalize{set1<VectorExtension, vector_base_t_granularity::value>( p_Normalize ) },
            m_Alignment{set1<VectorExtension, vector_base_t_granularity::value>(
               BiggestSupportedVectorExtension::vector_helper_t::element_count::value
            )},
            m_AlignmentHelper{set1<VectorExtension, vector_base_t_granularity::value>(
               BiggestSupportedVectorExtension::vector_helper_t::element_count::value - 1
            )},
            m_AlignmentHelperComplement{set1<VectorExtension, vector_base_t_granularity::value>(
               - BiggestSupportedVectorExtension::vector_helper_t::element_count::value
            )},
            m_Zero{set1<VectorExtension, vector_base_t_granularity::value>(
               0
            )} { }
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
         vector_t tmpResult =
            mul<VectorExtension>::apply(
               p_Key,
               p_State.m_Prime
            );
         add<VectorExtension>::apply(
            tmpResult,
            add<VectorExtension>::apply(
               p_State.m_Alignment,
               or<VectorExtension
            )
         )
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
