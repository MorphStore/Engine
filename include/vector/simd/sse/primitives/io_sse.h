/**
 * @file io.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_IO_SSE_H
#define MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_IO_SSE_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/sse/extension_sse.h>
#include <vector/primitives/io.h>

#include <functional>


namespace vector {


   template<typename T, int IOGranularity>
   struct io<sse<v128<T>>,iov::ALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned integer values into 128 Bit vector register." );
         return _mm_load_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned float values into 128 Bit vector register." );
         return _mm_load_ps(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned double values into 128 Bit vector register." );
         return _mm_load_pd(reinterpret_cast< U const * >(p_DataPtr));
      }
   };

   template<typename T, int IOGranularity>
   struct io<sse<v128<T>>,iov::STREAM, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Stream load integer values into 128 Bit vector register." );
         return _mm_stream_load_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
      }
   };

   template<typename T, int IOGranularity>
   struct io<sse<v128<T>>,iov::UNALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 128 Bit vector register." );
         return _mm_loadu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned float values into 128 Bit vector register." );
         return _mm_loadu_ps(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned double values into 128 Bit vector register." );
         return _mm_loadu_pd(reinterpret_cast< U const * >(p_DataPtr));
      }
   };

   template<typename T, int IOGranularity>
   struct io<sse<v128<T>>,iov::UNALIGNEDX, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 128 Bit vector register using lddqu." );
         return _mm_lddqu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
      }
   };

//   template<class VectorExtension, iov IOVariant, int IOGranularity>
//   std::function< typename VectorExtension::vector_t ( typename VectorExtension::base_t const * const ) > foo = io< VectorExtension, IOVariant, IOGranularity >::load;
}


#endif //MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_IO_SSE_H
