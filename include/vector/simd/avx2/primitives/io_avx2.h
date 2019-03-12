/**
 * @file io.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_IO_AVX2_H
#define MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_IO_AVX2_H

#include <core/utils/preprocessor.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/primitives/io.h>

namespace vector {
   template<typename T, int IOGranularity>
   struct load<avx2<v256<T>>,iov::ALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned integer values into 256 Bit vector register." );
         return _mm256_load_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t const *>(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned float values into 256 Bit vector register." );
         return _mm256_load_ps(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned double values into 256 Bit vector register." );
         return _mm256_load_pd(reinterpret_cast< U const * >(p_DataPtr));
      }
   };

   template<typename T, int IOGranularity>
   struct load<avx2<v256<T>>,iov::STREAM, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Stream load integer values into 256 Bit vector register." );
         return _mm256_stream_load_si128(reinterpret_cast<typename avx2< v256< U > >::vector_t const *>(p_DataPtr));
      }
   };

   template<typename T, int IOGranularity>
   struct load<avx2<v256<T>>,iov::UNALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 256 Bit vector register." );
         return _mm256_loadu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t const *>(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned float values into 256 Bit vector register." );
         return _mm256_loadu_ps(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned double values into 256 Bit vector register." );
         return _mm256_loadu_pd(reinterpret_cast< U const * >(p_DataPtr));
      }
   };

   template<typename T, int IOGranularity>
   struct load<avx2<v256<T>>,iov::UNALIGNEDX, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 256 Bit vector register using lddqu." );
         return _mm256_lddqu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t const *>(p_DataPtr));
      }
   };


}

#endif //MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_IO_AVX2_H