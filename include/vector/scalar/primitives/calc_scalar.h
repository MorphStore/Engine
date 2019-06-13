//
// Created by jpietrzyk on 09.05.19.
//

#ifndef MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_CALC_SCALAR_H
#define MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_CALC_SCALAR_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/primitives/calc.h>

#include <functional>

namespace vector{
   template<>
   struct add<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         typename scalar<v64<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 64 bit integer values from two registers (scalar)" );
         return p_vec1 + p_vec2;
      }
   };
   template<>
   struct sub<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         typename scalar<v64<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 64 bit integer values from two registers (scalar)" );
         return p_vec1 - p_vec2;
      }
   };
   template<>
   struct hadd<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::base_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Horizontally add (return value) 64 bit integer values one register (scalar)" );
         return p_vec1;
      }
   };
   template<>
   struct mul<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         typename scalar<v64<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (scalar)" );
         return p_vec1 * p_vec2;
      }
   };
   template<>
   struct div<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         typename scalar<v64<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Divide 64 bit integer values from two registers (scalar)" );

         return p_vec1 / p_vec2;
      }
   };
   template<>
   struct mod<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         typename scalar<v64<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Modulo divide 64 bit integer values from two registers (scalar)" );
         return p_vec1 % p_vec2;
      }
   };
   template<>
   struct inv<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 64 bit integer values of one register (scalar)" );
         return ((~p_vec1)+1);
      }
   };
   template<>
   struct shift_left<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (scalar)" );
         return p_vec1 << p_distance;
      }
   };
   template<>
   struct shift_right<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (scalar)" );
         return p_vec1 >> p_distance;
      }
   };
}
#endif //MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_CALC_SCALAR_H
