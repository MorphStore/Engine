//
// Created by jpietrzyk on 16.05.19.
//

#ifndef MORPHSTORE_SET_UTILS_H
#define MORPHSTORE_SET_UTILS_H


#include <vector/general_vector.h>
#include <core/utils/preprocessor.h>
#include <vector/primitives/calc.h>
#include <vector/primitives/create.h>
#include <vector/primitives/logic.h>
#include <core/utils/math.h>
namespace vector {


   enum class size_policy_set {
      ARBITRARY,
      EXPONENTIAL
   };

   template< int MaximumLoadFactor >
   constexpr size_t get_size_for_exponential_set( size_t const p_DataCount ) {
      size_t tmp = ( 1 << ( 64 - __builtin_clzl( p_DataCount ) ) );
      return ( ( tmp << ( ( tmp * 100 / MaximumLoadFactor ) >= p_DataCount ) ) - 1 );
   }

   template< int MaximumLoadFactor, size_policy_set SPS >
   struct key_size_helper;

   template< int MaximumLoadFactor >
   struct key_size_helper< MaximumLoadFactor, size_policy_set::ARBITRARY > {
      size_t const m_Count;
      key_size_helper(size_t const p_DataCount) :
         m_Count{p_DataCount * 100 / MaximumLoadFactor}{}
   };
   template< int MaximumLoadFactor >
   struct key_size_helper< MaximumLoadFactor, size_policy_set::EXPONENTIAL > {
      size_t const m_Count;
      key_size_helper(size_t const p_DataCount) :
         m_Count{get_size_for_exponential_set<MaximumLoadFactor>( p_DataCount )}{}
   };

   // Key resizer can be used to map an arbitrary index (resulting from a hash function) onto a finite index within
   // the target datastructure. We distinguish between exponential and arbitrary sized target datastructures.
   // To fit fit the index into the exponential sized structures ( 2 ^ x ) only a logical AND has to be performed.
   // If the size of the datastructure is arbitrary, a modulo Operator is called.
   template< class VectorExtension, size_policy_set SPS >
   struct key_resizer;

   template< class VectorExtension >
   struct key_resizer< VectorExtension, size_policy_set::ARBITRARY > {
      struct state_t {
         IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
         size_t const m_ResizeValue;
         vector_t const m_ResizeVector;
         state_t(size_t const p_Count ) :
            m_ResizeValue{ p_Count },
            m_ResizeVector{set1<VectorExtension>(p_Count)} {}
      };
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_t
      apply( vector_t const & p_Key, state_t const & p_State ) {
         return
            mod<VectorExtension>::apply(
               p_Key,
               p_State.m_ResizeVector
            );
      }
   };



   template< class VectorExtension >
   struct key_resizer< VectorExtension, size_policy_set::EXPONENTIAL > {
      struct state_t {
         IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
         size_t const m_ResizeValue;
         vector_t const m_ResizeVector;
         state_t(size_t const p_Count) :
            m_ResizeValue{ p_Count },
            m_ResizeVector{ set1<VectorExtension, vector_base_t_granularity::value>( p_Count ) } {}
      };
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_t
      apply( vector_t const & p_Key, state_t const & p_State ) {
         return
            logical_and<VectorExtension, vector_size_bit::value>(
               p_Key,
               p_State.m_ResizeVector
            );
      }
   };


   // Key aligner is used to 'align' every position to the biggest needed position:
   // If we hash our keys using avx2 vector registers, the key should point to a position, which not violates
   // the overall size constraints of the underlying data structure. Thus we 'normalize' the position in a way, that
   // the position is a multiple of the number of elements within the avx2 vector register.
   // Thus we want to be able to process the underlying set with different vector extensions, the extension providing
   // the biggest vector register has to be choosen as a reference for the described aligning.
   template< class VectorExtension >
   struct key_aligner {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      template< class BiggestSupportedVectorExtension >
      struct state_t {
         static_assert(
            std::is_same<
               typename VectorExtension::vector_helper_t::base_t,
               typename BiggestSupportedVectorExtension::vector_helper_t::base_t
            >::value,
            "Base type of specified vectorextension has to be equal to the base type of the biggest supported vector ext."
         );
         vector_t const m_AlignmentVector;
         state_t( void ) :
            m_AlignmentVector{
               set1<VectorExtension, VectorExtension::vector_helper_t::granularity::value>(
                  (base_t)~(BiggestSupportedVectorExtension::vector_helper_t::element_count::value - 1)
               )
            } {}
      };
      template< class BiggestSupportedVectorExtension >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_t
      apply( vector_t const & p_Key, state_t<BiggestSupportedVectorExtension> const & p_State ) {
         return logical_and<VectorExtension, vector_size_bit::value>( p_Key, p_State.m_AlignmentVector );
      }
   };


}

#endif //MORPHSTORE_SET_UTILS_H
