//
// Created by jpietrzyk on 09.05.19.
//

#ifndef MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_COMPARE_SCALAR_H
#define MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_COMPARE_SCALAR_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/primitives/compare.h>

#include <functional>

namespace vectorlib{
   template<>
   struct equal<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64<uint64_t>>::mask_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         typename scalar<v64<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: == ? (scalar)" );
         return (scalar<v64<uint64_t>>::vector_t)( p_vec1 == p_vec2 );
      }
   };
   template<>
   struct less<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64<uint64_t>>::mask_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         typename scalar<v64<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: < ? (scalar)" );
         return (scalar<v64<uint64_t>>::vector_t)( p_vec1 < p_vec2 );
      }
   };
   template<>
   struct lessequal<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64<uint64_t>>::mask_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         typename scalar<v64<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: <= ? (scalar)" );
         return (scalar<v64<uint64_t>>::vector_t)( p_vec1 <= p_vec2 );
      }
   };

   template<>
   struct greater<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64<uint64_t>>::mask_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         typename scalar<v64<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: > ? (scalar)" );
         return (scalar<v64<uint64_t>>::vector_t)( p_vec1 > p_vec2 );
      }
   };
   template<>
   struct greaterequal<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64<uint64_t>>::mask_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         typename scalar<v64<uint64_t>>::vector_t const & p_vec2
      ) {
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: >= ? (scalar)" );
         return (scalar<v64<uint64_t>>::vector_t)( p_vec1 >= p_vec2 );
      }
   };
   template<>
   struct count_matches<scalar<v64<uint64_t>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static uint8_t
      apply(
         typename scalar<v64<uint64_t>>::mask_t const & p_mask
      ) {
         trace( "[VECTOR] - Count matches in a comparison mask (scalar)" );
         return p_mask;
      }
   };


}

#endif /* MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_COMPARE_SCALAR_H */

