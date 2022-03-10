//
// Created by jpietrzyk on 28.05.19.
//

#ifndef MORPHSTORE_HASH_UTILS_H
#define MORPHSTORE_HASH_UTILS_H

#include <core/utils/preprocessor.h>

#include "../../vector_primitives.h"

#include <cstddef>
#include <type_traits>

namespace vectorlib {

   /**
    * Enum class to denote the size policy for a specific hash based datastructure.
    * Currently two types are distinguished:
    *   - ARBITRARY
    *       The hash based datastructure can have an arbitrary size ( greater or equal the number of elements which have
    *       to be inserted )
    *   - EXPONENTIAL
    *       The hash based datastructure consists of 2^n bucket, where 2^n is greater or equal to the number of elements
    *       which have to be inserted ).
    */
   enum class size_policy_hash {
      ARBITRARY,
      EXPONENTIAL
   };

   /**
    * Helper function to determine the number of buckets of the hash based datastructure if the size policy equals EXPONENTIAL.
    * @tparam MaximumLoadFactor The maximum load factor ( load factor = count( inserted_elements ) / count( all_buckets ).
    * Thus floats can not be a template parameter, the load factor is denoted using an int (60 means 0.6).
    * @param p_EstimatedDistinctCount The estimated filled bucket count.
    * @return A value denoting a size ( 2^n ) where 2^n >= ( p_EstimatedDistinctCount * 100 / MaximumLoadFactor ).
    */
   template< int MaximumLoadFactor >
   constexpr size_t get_size_for_exponential_set( size_t const p_EstimatedDistinctCount ) {
      size_t tmp = ( 1 << ( 64 - __builtin_clzl( p_EstimatedDistinctCount ) ) );
      return ( ( tmp << ( ( tmp * 100 / MaximumLoadFactor ) >= p_EstimatedDistinctCount ) ) );
   }

   /**
    * @brief Struct which provides the number of buckets for a hash based datastructure.
    * @detail The number of available buckets depend on an estimation of distinct values as well as a maximum loadfactor.
    * @tparam MaximumLoadFactor The maximum load factor ( load factor = count( inserted_elements ) / count( all_buckets ).
    * Thus floats can not be a template parameter, the load factor is denoted using an int (60 means 0.6).
    * @tparam SPH Either ARBITRARY or EXPONENTIAL.
    */
   template< class BiggestSupportedVectorExtension, int MaximumLoadFactor, size_policy_hash SPH >
   struct size_helper;
   template< class BiggestSupportedVectorExtension, int MaximumLoadFactor >
   struct size_helper< BiggestSupportedVectorExtension, MaximumLoadFactor, size_policy_hash::ARBITRARY > {
      size_t const m_UnalignedCount;
      size_t const m_Count;
      /**
       * Ctor. Sets the bucket count to p_EstimatedDistinctCount * 100 / MaximumLoadFactor.
       * @param p_EstimatedDistinctCount Number of estimated distinct values which should be inserted into a hash based datastructure.
       */
      size_helper(size_t const p_EstimatedDistinctCount) :
         m_UnalignedCount{
            p_EstimatedDistinctCount * 100 / MaximumLoadFactor
         },
         m_Count{
            ( m_UnalignedCount + BiggestSupportedVectorExtension::vector_helper_t::element_count::value +
               ( ( - ( m_UnalignedCount & ( BiggestSupportedVectorExtension::vector_helper_t::element_count::value - 1 ) ) )
                  | ( - BiggestSupportedVectorExtension::vector_helper_t::element_count::value )
               )
            )
         }{}
   };
   template< class BiggestSupportedVectorExtension, int MaximumLoadFactor >
   struct size_helper< BiggestSupportedVectorExtension, MaximumLoadFactor, size_policy_hash::EXPONENTIAL > {
      size_t const m_UnalignedCount;
      size_t const m_Count;
      /**
       * Ctor. Sets the bucket count to 2^n where 2^n >= ( p_EstimatedDistinctCount * 100 / MaximumLoadFactor ).
       * @param p_EstimatedDistinctCount Number of estimated distinct values which should be inserted into a hash based datastructure.
       */
      size_helper(size_t const p_DataCount) :
         m_UnalignedCount{get_size_for_exponential_set<MaximumLoadFactor>( p_DataCount )},
         m_Count{ m_UnalignedCount }{}
   };

   /**
    * @brief Struct which provides resize capabilities for positions within a hash based datastructure.
    * @details An existing hash based datastructure contains n buckets. If a key should be inserted a hash function
    * is applied onto that key to retrieve an arbitrary value wich is used as a positional hint.
    * This hint is used afterwards to look for either the key or an empty bucket. Naturally the hash value for a given
    * key can, if directly used as offsets on the underlying datastructure, exceed the limits of the datastructure. To
    * avoid segmentation violations the resulting offset has to be mapped into the finite index space of the target
    * datastructure. This can be done using the modulo operation with the size of the datastructure as operand.
    * @tparam VectorExtension Vector extension which is used for the operation.
    * @tparam SPS Size policy ( size_policy_hash::ARBITRARY | size_policy_hash::EXPONENTIAL )
    */
   template< class VectorExtension, size_policy_hash SPH >
   struct index_resizer;

   /**
    * @brief Index resizer for an index within an arbitrary sized hash based datastructure.
    * @tparam VectorExtension Vector extension which is used for the operation.
    */
   template< class VectorExtension >
   struct index_resizer< VectorExtension, size_policy_hash::ARBITRARY > {
      /**
       * @brief Helper struct which contains the needed values and vector registers for the resize operation.
       * @details Provides the actual modulo operand as a scalar value (size_t) as well as a vector register.
       */
      struct state_t {
         IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
         size_t const m_ResizeValue;
         vector_t const m_ResizeVector;
         state_t(size_t const p_Count ) :
            m_ResizeValue{ p_Count },
            m_ResizeVector{set1<VectorExtension, vector_base_t_granularity::value>(p_Count)} {}
      };
      /**
       * @brief Maps all keys from the vector register into the finite index space of the target hash based datastructure.
       * @param p_IndexValues Vector register which contains the indices to map.
       * @param p_State A state providing the needed modulo operand vector register.
       * @return A vector register containing the mapped indices.
       */
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_t
      apply( vector_t const & p_IndexValues, state_t const & p_State ) {
         return
            mod<VectorExtension>::apply(
               p_IndexValues,
               p_State.m_ResizeVector
            );
      }
   };
   /**
    * @brief Index resizer for an index within an exponential sized hash based datastructure.
    * @tparam VectorExtension Vector extension which is used for the operation.
    */
   template< class VectorExtension >
   struct index_resizer< VectorExtension, size_policy_hash::EXPONENTIAL > {
      /**
       * @brief Helper struct which contains the needed values and vector registers for the resize operation.
       * @details Provides the actual modulo operand as a scalar value (size_t) as well as a vector register. Thus the
       * modulo operation is substituted whith a logical AND, the operand is reduced by one. ( x % 2^n === x & (2^n - 1 ) ).
       */
      struct state_t {
         IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
         size_t const m_ResizeValue;
         vector_t const m_ResizeVector;
         state_t(size_t const p_Count) :
            m_ResizeValue{ p_Count - 1 },
            m_ResizeVector{ set1<VectorExtension, vector_base_t_granularity::value>( m_ResizeValue ) } {}
      };
      /**
       * @brief Maps all keys from the vector register into the finite index space of the target hash based datastructure.
       * @param p_IndexValues Vector register which contains the indices to map.
       * @param p_State A state providing the needed modulo operand vector register.
       * @return A vector register containing the mapped indices.
       */
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_t
      apply( vector_t const & p_IndexValues, state_t const & p_State ) {
         return
            bitwise_and<VectorExtension, vector_size_bit::value>(
               p_IndexValues,
               p_State.m_ResizeVector
            );
      }
   };

   /**
    * @brief Helper struct to align an index according to the biggest available vector extension.
    * @detail Given a hash based datastructure, the underlying container has a finite number of buckets. When searching
    * for an empty bucket, the index wich is used to identify a certain bucket has to be in the interval [0, $n$-1] where
    * $n$ is the amount of buckets within the container. When the linear search is vectorized, $v$ buckets are transferred
    * into a vector register and matched with the active key, where $v$ is the number of elements per vector register
    * ( SSE     [ uint64_t: v=2; uint32_t: v=4 ],
    *   AVX/2   [ uint64_t: v=4; uint32_t: v=8 ],
    *   AVX512  [ uint64_t: v=8; uint32_t: v=16 ] ).
    * When transferring $v$ consecutive buckets into a vector register starting at index $i$, $i$+$v$ has to be smaller
    * than $n$. The helper struct index_aligner ensures this by normalizing the index ($i'$) in a way that $i'$ is
    * divisible without remainder by $v$ of the biggest vector extension.
    * @tparam VectorExtension Vector extension which is used for linear search.
    */
   template< class VectorExtension >
   struct index_aligner {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      /**
       * @brief Helper struct for index_aligner.
       * @details Provides a vector register (VectorExtension::vector_t) containing the element count of the
       * BiggestSupportedVectorExtension decremented by one.
       * @tparam BiggestSupportedVectorExtension Biggest vector extension the linear search should be able to work with.
       */
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
      /**
       * @brief Tranforms all elements in p_IndexValues.
       * @details Transformation is done in a way, that all values within p_IndexValues are divisible without remainder by
       * the number of elements within BiggestSupportedVectorExtension.
       * @tparam BiggestSupportedVectorExtension Biggest vector extension the linear search should be able to work with.
       * @param p_IndexValues Constant reference to a vector register (VectorExtension::vector_t) which holds the indices.
       * @param p_State Constant reference to the helper struct for index_aligner (containing the transformation vector register).
       * @return Vector register (VectorExtension::vector_t) with transformed indices.
       */
      template< class BiggestSupportedVectorExtension >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_t
      apply( vector_t const & p_IndexValues, state_t<BiggestSupportedVectorExtension> const & p_State ) {
         return bitwise_and<VectorExtension, vector_size_bit::value>( p_IndexValues, p_State.m_AlignmentVector );
      }
   };

}
#endif //MORPHSTORE_HASH_UTILS_H
