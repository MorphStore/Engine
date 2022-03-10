/**
 * @file io_tsubasa.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_IO_TSUBASA_H
#define MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_IO_TSUBASA_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include "../../../simd/sse/extension_sse.h"
#include "../../../primitives/io.h"


namespace vectorlib {


   template<typename T, int IOGranularity>
   struct load<sse<v128<T>>,iov::ALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned integer values into 128 Bit vector register." );
         return _mm_load_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned float values into 128 Bit vector register." );
         return _mm_load_ps(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned double values into 128 Bit vector register." );
         return _mm_load_pd(reinterpret_cast< U const * >(p_DataPtr));
      }
   };

   template<typename T, int IOGranularity>
   struct load<sse<v128<T>>,iov::STREAM, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Stream load integer values into 128 Bit vector register." );
         return _mm_stream_load_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
      }
   };

   template<typename T, int IOGranularity>
   struct load<sse<v128<T>>,iov::UNALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 128 Bit vector register." );
         return _mm_loadu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned float values into 128 Bit vector register." );
         return _mm_loadu_ps(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned double values into 128 Bit vector register." );
         return _mm_loadu_pd(reinterpret_cast< U const * >(p_DataPtr));
      }
   };

   template<typename T, int IOGranularity>
   struct load<sse<v128<T>>,iov::UNALIGNEDX, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename sse< v128< U > >::vector_t
      apply( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 128 Bit vector register using lddqu." );
         return _mm_lddqu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
      }
   };



}

#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_IO_TSUBASA_H
