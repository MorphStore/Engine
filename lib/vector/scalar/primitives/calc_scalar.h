//
// Created by jpietrzyk on 09.05.19.
//

#ifndef MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_CALC_SCALAR_H
#define MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_CALC_SCALAR_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include "../extension_scalar.h"
#include "../../primitives/calc.h"
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
   
   /*NOTE: This primitive automatically substracts the unused bits, where a bitmask is larger than required*/
   template<>
   struct count_leading_zero<scalar<v64<uint64_t>>> {
     // template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static uint8_t
      apply(
         typename scalar<v64<uint64_t>>::mask_t const & p_mask
      ) {

         //return __builtin_clz(p_mask)-(sizeof(p_mask)*8-scalar<v64<U>>::vector_element_count::value);
          return __builtin_clz(p_mask)-63;
      }
   };
}
#endif //MORPHSTORE_VECTOR_SCALAR_PRIMITIVE_CALC_SCALAR_H
