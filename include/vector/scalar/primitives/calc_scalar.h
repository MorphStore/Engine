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
   template<>
   struct add<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         typename scalar<v64<uint64_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Add 64 bit integer values from two registers (scalar)" );
         return p_vec1 + p_vec2;
      }
   };

   template<>
   struct min<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         typename scalar<v64<uint64_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - build minimum of 64 bit integer values from two registers (sse)" );
         return std::min(p_vec1,p_vec2);
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
         TALLY_CALC_BINARY_SCALAR
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
         TALLY_CALC_UNARY_SCALAR
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
         TALLY_CALC_BINARY_SCALAR
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
         TALLY_CALC_BINARY_SCALAR
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
         TALLY_CALC_BINARY_SCALAR
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
         TALLY_CALC_UNARY_SCALAR
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
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (all by the same distance) (scalar)" );
         return p_vec1 << p_distance;
      }
   };
   template<>
   struct shift_left_individual<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_data,
         typename scalar<v64<uint64_t>>::vector_t const & p_distance
      ){
         TALLY_CALC_BINARY_SCALAR
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
   template<>
   struct shift_right<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (all by the same distance) (scalar)" );
         return p_vec1 >> p_distance;
      }
   };
   template<>
   struct shift_right_individual<scalar<v64<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<uint64_t>>::vector_t
      apply(
         typename scalar<v64<uint64_t>>::vector_t const & p_data,
         typename scalar<v64<uint64_t>>::vector_t const & p_distance
      ){
         TALLY_CALC_BINARY_SCALAR
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

   //start 32 bit

   template<>
   struct add<scalar<v32<uint32_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v32<uint32_t>>::vector_t
      apply(
         typename scalar<v32<uint32_t>>::vector_t const & p_vec1,
         typename scalar<v32<uint32_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Add 64 bit integer values from two registers (scalar)" );
         return p_vec1 + p_vec2;
      }
   };
   template<>
   struct min<scalar<v32<uint32_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v32<uint32_t>>::vector_t
      apply(
         typename scalar<v32<uint32_t>>::vector_t const & p_vec1,
         typename scalar<v32<uint32_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - build minimum of 64 bit integer values from two registers (sse)" );
         return std::min(p_vec1,p_vec2);
      }
   };
   template<>
   struct sub<scalar<v32<uint32_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v32<uint32_t>>::vector_t
      apply(
         typename scalar<v32<uint32_t>>::vector_t const & p_vec1,
         typename scalar<v32<uint32_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Subtract 64 bit integer values from two registers (scalar)" );
         return p_vec1 - p_vec2;
      }
   };
   template<>
   struct hadd<scalar<v32<uint32_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v32<uint32_t>>::base_t
      apply(
         typename scalar<v32<uint32_t>>::vector_t const & p_vec1
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Horizontally add (return value) 64 bit integer values one register (scalar)" );
         return p_vec1;
      }
   };
   template<>
   struct mul<scalar<v32<uint32_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v32<uint32_t>>::vector_t
      apply(
         typename scalar<v32<uint32_t>>::vector_t const & p_vec1,
         typename scalar<v32<uint32_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (scalar)" );
         return p_vec1 * p_vec2;
      }
   };
   template<>
   struct div<scalar<v32<uint32_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v32<uint32_t>>::vector_t
      apply(
         typename scalar<v32<uint32_t>>::vector_t const & p_vec1,
         typename scalar<v32<uint32_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Divide 64 bit integer values from two registers (scalar)" );

         return p_vec1 / p_vec2;
      }
   };
   template<>
   struct mod<scalar<v32<uint32_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v32<uint32_t>>::vector_t
      apply(
         typename scalar<v32<uint32_t>>::vector_t const & p_vec1,
         typename scalar<v32<uint32_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Modulo divide 64 bit integer values from two registers (scalar)" );
         return p_vec1 % p_vec2;
      }
   };
   template<>
   struct inv<scalar<v32<uint32_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v32<uint32_t>>::vector_t
      apply(
         typename scalar<v32<uint32_t>>::vector_t const & p_vec1
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Additive inverting 64 bit integer values of one register (scalar)" );
         return ((~p_vec1)+1);
      }
   };
   template<>
   struct shift_left<scalar<v32<uint32_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v32<uint32_t>>::vector_t
      apply(
         typename scalar<v32<uint32_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (all by the same distance) (scalar)" );
         return p_vec1 << p_distance;
      }
   };
   template<>
   struct shift_left_individual<scalar<v32<uint32_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v32<uint32_t>>::vector_t
      apply(
         typename scalar<v32<uint32_t>>::vector_t const & p_data,
         typename scalar<v32<uint32_t>>::vector_t const & p_distance
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (each by its individual distance) (scalar)" );
         // The scalar shift does not do anything when the distance is the
         // number of digits.
         // @todo Currently, this is a workaround, rethink whether we want it
         // this way and whether shift_left above should do it the same way.
         // CHANGED std::numeric_limits<scalar<v64<uint64_t>> to std::numeric_limits<scalar<v32<uint32_t>>
         if(p_distance == std::numeric_limits<scalar<v32<uint32_t>>::vector_t>::digits)
             return 0;
         else
             return p_data << p_distance;
      }
   };
   template<>
   struct shift_right<scalar<v32<uint32_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v32<uint32_t>>::vector_t
      apply(
         typename scalar<v32<uint32_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (all by the same distance) (scalar)" );
         return p_vec1 >> p_distance;
      }
   };
   template<>
   struct shift_right_individual<scalar<v32<uint32_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v32<uint32_t>>::vector_t
      apply(
         typename scalar<v32<uint32_t>>::vector_t const & p_data,
         typename scalar<v32<uint32_t>>::vector_t const & p_distance
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (each by its individual distance) (scalar)" );
         // The scalar shift does not do anything when the distance is the
         // number of digits.
         // @todo Currently, this is a workaround, rethink whether we want it
         // this way and whether shift_left above should do it the same way.
         // CHANGED std::numeric_limits<scalar<v64<uint64_t>> to std::numeric_limits<scalar<v32<uint32_t>>
         if(p_distance == std::numeric_limits<scalar<v32<uint32_t>>::vector_t>::digits)
             return 0;
         else
             return p_data >> p_distance;
      }
   };

   //start 16 bit

   template<>
   struct add<scalar<v16<uint16_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v16<uint16_t>>::vector_t
      apply(
         typename scalar<v16<uint16_t>>::vector_t const & p_vec1,
         typename scalar<v16<uint16_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Add 64 bit integer values from two registers (scalar)" );
         return p_vec1 + p_vec2;
      }
   };
   template<>
   struct min<scalar<v16<uint16_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v16<uint16_t>>::vector_t
      apply(
         typename scalar<v16<uint16_t>>::vector_t const & p_vec1,
         typename scalar<v16<uint16_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - build minimum of 64 bit integer values from two registers (sse)" );
         return std::min(p_vec1,p_vec2);
      }
   };
   template<>
   struct sub<scalar<v16<uint16_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v16<uint16_t>>::vector_t
      apply(
         typename scalar<v16<uint16_t>>::vector_t const & p_vec1,
         typename scalar<v16<uint16_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Subtract 64 bit integer values from two registers (scalar)" );
         return p_vec1 - p_vec2;
      }
   };
   template<>
   struct hadd<scalar<v16<uint16_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v16<uint16_t>>::base_t
      apply(
         typename scalar<v16<uint16_t>>::vector_t const & p_vec1
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Horizontally add (return value) 64 bit integer values one register (scalar)" );
         return p_vec1;
      }
   };
   template<>
   struct mul<scalar<v16<uint16_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v16<uint16_t>>::vector_t
      apply(
         typename scalar<v16<uint16_t>>::vector_t const & p_vec1,
         typename scalar<v16<uint16_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (scalar)" );
         return p_vec1 * p_vec2;
      }
   };
   template<>
   struct div<scalar<v16<uint16_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v16<uint16_t>>::vector_t
      apply(
         typename scalar<v16<uint16_t>>::vector_t const & p_vec1,
         typename scalar<v16<uint16_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Divide 64 bit integer values from two registers (scalar)" );

         return p_vec1 / p_vec2;
      }
   };
   template<>
   struct mod<scalar<v16<uint16_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v16<uint16_t>>::vector_t
      apply(
         typename scalar<v16<uint16_t>>::vector_t const & p_vec1,
         typename scalar<v16<uint16_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Modulo divide 64 bit integer values from two registers (scalar)" );
         return p_vec1 % p_vec2;
      }
   };
   template<>
   struct inv<scalar<v16<uint16_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v16<uint16_t>>::vector_t
      apply(
         typename scalar<v16<uint16_t>>::vector_t const & p_vec1
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Additive inverting 64 bit integer values of one register (scalar)" );
         return ((~p_vec1)+1);
      }
   };
   template<>
   struct shift_left<scalar<v16<uint16_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v16<uint16_t>>::vector_t
      apply(
         typename scalar<v16<uint16_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (all by the same distance) (scalar)" );
         return p_vec1 << p_distance;
      }
   };
   template<>
   struct shift_left_individual<scalar<v16<uint16_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v16<uint16_t>>::vector_t
      apply(
         typename scalar<v16<uint16_t>>::vector_t const & p_data,
         typename scalar<v16<uint16_t>>::vector_t const & p_distance
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (each by its individual distance) (scalar)" );
         // The scalar shift does not do anything when the distance is the
         // number of digits.
         // @todo Currently, this is a workaround, rethink whether we want it
         // this way and whether shift_left above should do it the same way.
         // CHANGED std::numeric_limits<scalar<v64<uint64_t>> to std::numeric_limits<scalar<v16<uint16_t>>
         if(p_distance == std::numeric_limits<scalar<v16<uint16_t>>::vector_t>::digits)
             return 0;
         else
             return p_data << p_distance;
      }
   };
   template<>
   struct shift_right<scalar<v16<uint16_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v16<uint16_t>>::vector_t
      apply(
         typename scalar<v16<uint16_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (all by the same distance) (scalar)" );
         return p_vec1 >> p_distance;
      }
   };
   template<>
   struct shift_right_individual<scalar<v16<uint16_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v16<uint16_t>>::vector_t
      apply(
         typename scalar<v16<uint16_t>>::vector_t const & p_data,
         typename scalar<v16<uint16_t>>::vector_t const & p_distance
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (each by its individual distance) (scalar)" );
         // The scalar shift does not do anything when the distance is the
         // number of digits.
         // @todo Currently, this is a workaround, rethink whether we want it
         // this way and whether shift_left above should do it the same way.
         // CHANGED std::numeric_limits<scalar<v64<uint64_t>> to std::numeric_limits<scalar<v32<uint32_t>>
         if(p_distance == std::numeric_limits<scalar<v16<uint16_t>>::vector_t>::digits)
             return 0;
         else
             return p_data >> p_distance;
      }
   };

   //start 8 bit

   template<>
   struct add<scalar<v8<uint8_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v8<uint8_t>>::vector_t
      apply(
         typename scalar<v8<uint8_t>>::vector_t const & p_vec1,
         typename scalar<v8<uint8_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Add 64 bit integer values from two registers (scalar)" );
         return p_vec1 + p_vec2;
      }
   };
   template<>
   struct min<scalar<v8<uint8_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v8<uint8_t>>::vector_t
      apply(
         typename scalar<v8<uint8_t>>::vector_t const & p_vec1,
         typename scalar<v8<uint8_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - build minimum of 64 bit integer values from two registers (sse)" );
         return std::min(p_vec1,p_vec2);
      }
   };
   template<>
   struct sub<scalar<v8<uint8_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v8<uint8_t>>::vector_t
      apply(
         typename scalar<v8<uint8_t>>::vector_t const & p_vec1,
         typename scalar<v8<uint8_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Subtract 64 bit integer values from two registers (scalar)" );
         return p_vec1 - p_vec2;
      }
   };
   template<>
   struct hadd<scalar<v8<uint8_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v8<uint8_t>>::base_t
      apply(
         typename scalar<v8<uint8_t>>::vector_t const & p_vec1
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Horizontally add (return value) 64 bit integer values one register (scalar)" );
         return p_vec1;
      }
   };
   template<>
   struct mul<scalar<v8<uint8_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v8<uint8_t>>::vector_t
      apply(
         typename scalar<v8<uint8_t>>::vector_t const & p_vec1,
         typename scalar<v8<uint8_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (scalar)" );
         return p_vec1 * p_vec2;
      }
   };
   template<>
   struct div<scalar<v8<uint8_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v8<uint8_t>>::vector_t
      apply(
         typename scalar<v8<uint8_t>>::vector_t const & p_vec1,
         typename scalar<v8<uint8_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Divide 64 bit integer values from two registers (scalar)" );

         return p_vec1 / p_vec2;
      }
   };
   template<>
   struct mod<scalar<v8<uint8_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v8<uint8_t>>::vector_t
      apply(
         typename scalar<v8<uint8_t>>::vector_t const & p_vec1,
         typename scalar<v8<uint8_t>>::vector_t const & p_vec2
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Modulo divide 64 bit integer values from two registers (scalar)" );
         return p_vec1 % p_vec2;
      }
   };
   template<>
   struct inv<scalar<v8<uint8_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v8<uint8_t>>::vector_t
      apply(
         typename scalar<v8<uint8_t>>::vector_t const & p_vec1
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Additive inverting 64 bit integer values of one register (scalar)" );
         return ((~p_vec1)+1);
      }
   };
   template<>
   struct shift_left<scalar<v8<uint8_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v8<uint8_t>>::vector_t
      apply(
         typename scalar<v8<uint8_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (all by the same distance) (scalar)" );
         return p_vec1 << p_distance;
      }
   };
   template<>
   struct shift_left_individual<scalar<v8<uint8_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v8<uint8_t>>::vector_t
      apply(
         typename scalar<v8<uint8_t>>::vector_t const & p_data,
         typename scalar<v8<uint8_t>>::vector_t const & p_distance
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (each by its individual distance) (scalar)" );
         // The scalar shift does not do anything when the distance is the
         // number of digits.
         // @todo Currently, this is a workaround, rethink whether we want it
         // this way and whether shift_left above should do it the same way.
         // CHANGED std::numeric_limits<scalar<v64<uint64_t>> to std::numeric_limits<scalar<v8<uint8_t>>
         if(p_distance == std::numeric_limits<scalar<v8<uint8_t>>::vector_t>::digits)
             return 0;
         else
             return p_data << p_distance;
      }
   };
   template<>
   struct shift_right<scalar<v8<uint8_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v8<uint8_t>>::vector_t
      apply(
         typename scalar<v8<uint8_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         TALLY_CALC_UNARY_SCALAR
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (all by the same distance) (scalar)" );
         return p_vec1 >> p_distance;
      }
   };
   template<>
   struct shift_right_individual<scalar<v8<uint8_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v8<uint8_t>>::vector_t
      apply(
         typename scalar<v8<uint8_t>>::vector_t const & p_data,
         typename scalar<v8<uint8_t>>::vector_t const & p_distance
      ){
         TALLY_CALC_BINARY_SCALAR
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (each by its individual distance) (scalar)" );
         // The scalar shift does not do anything when the distance is the
         // number of digits.
         // @todo Currently, this is a workaround, rethink whether we want it
         // this way and whether shift_left above should do it the same way.
         // CHANGED std::numeric_limits<scalar<v64<uint64_t>> to std::numeric_limits<scalar<v8<uint8_t>>
         if(p_distance == std::numeric_limits<scalar<v8<uint8_t>>::vector_t>::digits)
             return 0;
         else
             return p_data >> p_distance;
      }
   };
}
#endif //MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_CALC_SCALAR_H
