//
// Created by jpietrzyk on 20.05.19.
//

#ifndef MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_LOGIC_TSUBASA_H
#define MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_LOGIC_TSUBASA_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/vecprocessor/tsubasa/extension_tsubasa.h>
#include <vector/primitives/logic.h>


namespace vectorlib {


   template<>
   struct logic< aurora< v16k< uint64_t > >, aurora< v16k< uint64_t > >::vector_helper_t::size_bit::value > {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename aurora< v16k< uint64_t > >::vector_t
      bitwise_and(
         typename aurora< v16k< uint64_t > >::vector_t const & p_Vec1,
         typename aurora< v16k< uint64_t > >::vector_t const & p_Vec2
      ) {
         return _ve_vand_vvv( p_Vec1, p_Vec2 );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename aurora< v16k< uint64_t > >::mask_t
      bitwise_and(
         typename aurora< v16k< uint64_t > >::mask_t p_Vec1,
         typename aurora< v16k< uint64_t > >::mask_t p_Vec2
      ) {
         return _ve_andm_mmm( p_Vec1, p_Vec2 );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename aurora< v16k< uint64_t > >::vector_t
      bitwise_or(
         typename aurora< v16k< uint64_t > >::vector_t const & p_Vec1,
         typename aurora< v16k< uint64_t > >::vector_t const & p_Vec2
      ) {
         return _ve_vor_vvv( p_Vec1, p_Vec2 );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename aurora< v16k< uint64_t > >::mask_t
      bitwise_and(
         typename aurora< v16k< uint64_t > >::mask_t p_Vec1,
         typename aurora< v16k< uint64_t > >::mask_t p_Vec2
      ) {
         return _ve_orm_mmm( p_Vec1, p_Vec2 );
      }
   };


}
#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_LOGIC_TSUBASA_H
