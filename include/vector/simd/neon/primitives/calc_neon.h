/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   calc_neon.h
 * Author: Annett
 *
 * Created on 1. August 2019, 15:56
 */

#ifndef CALC_NEON_H
#define CALC_NEON_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/neon/extension_neon.h>
#include <vector/primitives/calc.h>

#include <functional>

namespace vectorlib{

   template<>
   struct add<neon<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename neon<v128<uint64_t>>::vector_t
      apply(
         typename neon<v128<uint64_t>>::vector_t const & p_vec1,
         typename neon<v128<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 64 bit integer values from two registers (neon)" );
         return vaddq_u64( p_vec1, p_vec2);
      }
   };
   

   template<>
   struct min<neon<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename neon<v128<uint64_t>>::vector_t
      apply(
         typename neon<v128<uint64_t>>::vector_t const & p_vec1,
         typename neon<v128<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Add 64 bit integer values from two registers (neon)" );
         return vbslq_u64(vcltq_u64(p_vec1, p_vec2),p_vec1, p_vec2);
      }
   };
   
   template<>
   struct sub<neon<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename neon<v128<uint64_t>>::vector_t
      apply(
         typename neon<v128<uint64_t>>::vector_t const & p_vec1,
         typename neon<v128<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Subtract 64 bit integer values from two registers (neon)" );
         return vsubq_u64( p_vec1, p_vec2);
      }
   };
   template<>
   struct hadd<neon<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename neon<v128<uint64_t>>::base_t
      apply(
         typename neon<v128<uint64_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Horizontally add 64 bit integer values one register (neon)" );
         return
             vgetq_lane_u64(p_vec1, 0) +  vgetq_lane_u64(p_vec1, 1);
      }
   };
   
   template<>
   struct mul<neon<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename neon<v128<uint64_t>>::vector_t
      apply(
         typename neon<v128<uint64_t>>::vector_t const & p_vec1,
         typename neon<v128<uint64_t>>::vector_t const & p_vec2
      ){
         trace( "[VECTOR] - Multiply 64 bit integer values from two registers (neon)" );
         info( "[VECTOR] - vmovn_u64 is called before multiplying -> only the lower 32 bits are processed" );
         return vmull_u32( vmovn_u64(p_vec1), vmovn_u64(p_vec2)); //TODO Does this really work?
      }
   };

  
   template<>
   struct inv<neon<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename neon<v128<uint64_t>>::vector_t
      apply(
         typename neon<v128<uint64_t>>::vector_t const & p_vec1
      ){
         trace( "[VECTOR] - Additive inverting 64 bit integer values of one register (neon)" );
         return vsubq_u64( vdupq_n_u64(0), p_vec1);
      }
   };

   template<>
   struct shift_left<neon<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename neon<v128<uint64_t>>::vector_t
      apply(
         typename neon<v128<uint64_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (all by the same distance) (neon)" );
         return vshlq_n_u64(p_vec1, p_distance);
      }
   };

   template<>
   struct shift_left_individual<neon<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename neon<v128<uint64_t>>::vector_t
      apply(
         typename neon<v128<uint64_t>>::vector_t const & p_data,
         typename neon<v128<uint64_t>>::vector_t const & p_distance
      ){
         
         trace( "[VECTOR] - Left-shifting 64 bit integer values of one register (each by its individual distance) (neon)" );
         return vshlq_u64(p_data,p_distance);
      }
   };

   template<>
   struct shift_right<neon<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename neon<v128<uint64_t>>::vector_t
      apply(
         typename neon<v128<uint64_t>>::vector_t const & p_vec1,
         int const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (all by the same distance) (neon)" );
         return vshrq_n_u64(p_vec1, p_distance);
      }
   };

   template<>
   struct shift_right_individual<neon<v128<uint64_t>>/*, 64*/> {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename neon<v128<uint64_t>>::vector_t
      apply(
         typename neon<v128<uint64_t>>::vector_t const & p_data,
         typename neon<v128<uint64_t>>::vector_t const & p_distance
      ){
         trace( "[VECTOR] - Right-shifting 64 bit integer values of one register (each by its individual distance) (neon)" );
         return vshlq_u64(p_data, vsubq_u64( vdupq_n_u64(0), p_distance));
      }
   };

}

#endif /* CALC_NEON_H */

