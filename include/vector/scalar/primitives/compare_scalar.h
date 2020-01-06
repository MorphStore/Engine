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
    template<typename T>
   struct equal<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64<T>>::mask_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         typename scalar<v64<T>>::vector_t const & p_vec2
      ) {
#if tally
compare_scalar += 1;
#endif
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: == ? (scalar)" );
         return (typename scalar<v64<uint64_t>>::mask_t)( p_vec1 == p_vec2 );
      }
   };
    template<typename T>
   struct less<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64<T>>::mask_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         typename scalar<v64<T>>::vector_t const & p_vec2
      ) {
#if tally
compare_scalar += 1;
#endif
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: < ? (scalar)" );
         return (typename scalar<v64<T>>::mask_t)( p_vec1 < p_vec2 );
      }
   };
    template<typename T>
   struct lessequal<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64<T>>::mask_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         typename scalar<v64<T>>::vector_t const & p_vec2
      ) {
#if tally
compare_scalar += 1;
#endif
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: <= ? (scalar)" );
         return (typename scalar<v64<uint64_t>>::mask_t)( p_vec1 <= p_vec2 );
      }
   };

    template<typename T>
   struct greater<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64<T>>::mask_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         typename scalar<v64<T>>::vector_t const & p_vec2
      ) {
#if tally
compare_scalar += 1;
#endif
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: > ? (scalar)" );
         return (typename scalar<v64<T>>::mask_t)( p_vec1 > p_vec2 );
      }
   };
    template<typename T>
   struct greaterequal<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64<T>>::mask_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         typename scalar<v64<T>>::vector_t const & p_vec2
      ) {
#if tally
compare_scalar += 1;
#endif
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: >= ? (scalar)" );
         return (typename scalar<v64<T>>::mask_t)( p_vec1 >= p_vec2 );
      }
   };
    template<typename T>
   struct count_matches<scalar<v64<T>>> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static uint8_t
      apply(
         typename scalar<v64<T>>::mask_t const & p_mask
      ) {
#if tally
compare_scalar += 1;
#endif
         trace( "[VECTOR] - Count matches in a comparison mask (scalar)" );
         return p_mask;
      }
   };


}

#endif /* MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_COMPARE_SCALAR_H */
