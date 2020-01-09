//
// Created by jpietrzyk on 09.05.19.
//

#ifndef MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_CALC_SCALAR_H
#define MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_CALC_SCALAR_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/primitives/calc.h>
#include <algorithm>

#include <functional>
#include <limits>

namespace vectorlib{
   template<typename T>
   struct add<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         typename scalar<v64<T>>::vector_t const & p_vec2
      ){
#if tally
calc_binary_scalar += 1;
#endif
         trace( "[VECTOR] - Add 64 bit integer values from two registers (scalar)" );
         return p_vec1 + p_vec2;
      }
   };

   template<typename T>
   struct min<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         typename scalar<v64<T>>::vector_t const & p_vec2
      ){
#if tally
calc_binary_scalar += 1;
#endif
         trace( "[VECTOR] - build minimum of 64 bit integer values from two registers (sse)" );
         return std::min(p_vec1,p_vec2);
      }
   };

   template<typename T>
   struct sub<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         typename scalar<v64<T>>::vector_t const & p_vec2
      ){
#if tally
calc_binary_scalar += 1;
#endif
         trace( "[VECTOR] - Subtract 64 bit integer values from two registers (scalar)" );
         return p_vec1 - p_vec2;
      }
   };
    template<typename T>
   struct hadd<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::base_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1
      ){
#if tally
calc_unary_scalar += 1;
#endif
         trace( "[VECTOR] - Horizontally add (return value) 64 bit integer values one register (scalar)" );
         return p_vec1;
      }
   };
    template<typename T>
   struct mul<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         typename scalar<v64<T>>::vector_t const & p_vec2
      ){
#if tally
calc_binary_scalar += 1;
#endif
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (scalar)" );
         return p_vec1 * p_vec2;
      }
   };
    template<typename T>
   struct div<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         typename scalar<v64<T>>::vector_t const & p_vec2
      ){
#if tally
calc_binary_scalar += 1;
#endif
         trace( "[VECTOR] - Divide 64 bit integer values from two registers (scalar)" );

         return p_vec1 / p_vec2;
      }
   };
    template<typename T>
   struct mod<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         typename scalar<v64<T>>::vector_t const & p_vec2
      ){
#if tally
calc_binary_scalar += 1;
#endif
         trace( "[VECTOR] - Modulo divide 64 bit integer values from two registers (scalar)" );
         return p_vec1 % p_vec2;
      }
   };
    template<typename T>
   struct inv<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1
      ){
#if tally
calc_unary_scalar += 1;
#endif
         trace( "[VECTOR] - Additive inverting 64 bit integer values of one register (scalar)" );
         return ((~p_vec1)+1);
      }
   };
    template<typename T>
   struct shift_left<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         int const & p_distance
      ){
#if tally
calc_binary_scalar += 1;      //HELP SAME CASE AS AVX2
#endif
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (all by the same distance) (scalar)" );
         return p_vec1 << p_distance;
      }
   };
    template<typename T>
   struct shift_left_individual<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_data,
         typename scalar<v64<T>>::vector_t const & p_distance
      ){
#if tally
calc_binary_scalar += 1;
#endif
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (each by its individual distance) (scalar)" );
         // The scalar shift does not do anything when the distance is the
         // number of digits.
         // @todo Currently, this is a workaround, rethink whether we want it
         // this way and whether shift_left above should do it the same way.
         if(p_distance == std::numeric_limits<scalar<v64<uint64_t>>::vector_t>::digits)
             return 0;
         else
             return p_data << p_distance;
      }
   };
    template<typename T>
   struct shift_right<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_vec1,
         int const & p_distance
      ){
#if tally
calc_binary_scalar += 1;         //HELP
#endif
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (all by the same distance) (scalar)" );
         return p_vec1 >> p_distance;
      }
   };
    template<typename T>
   struct shift_right_individual<scalar<v64<T>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      apply(
         typename scalar<v64<T>>::vector_t const & p_data,
         typename scalar<v64<T>>::vector_t const & p_distance
      ){
#if tally
calc_binary_scalar += 1;
#endif
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (each by its individual distance) (scalar)" );
         // The scalar shift does not do anything when the distance is the
         // number of digits.
         // @todo Currently, this is a workaround, rethink whether we want it
         // this way and whether shift_left above should do it the same way.
         if(p_distance == std::numeric_limits<scalar<v64<uint64_t>>::vector_t>::digits)
             return 0;
         else
             return p_data >> p_distance;
      }
   };
}
#endif //MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_CALC_SCALAR_H
