/**
 * @file io.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_IO_AVX2_H
#define MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_IO_AVX2_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/primitives/io.h>

#include <functional>

namespace vector {
    
    
   template<typename T, int IOGranularity>
   struct io<avx2<v256<T>>,iov::ALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned integer values into 256 Bit vector register." );
         return _mm256_load_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t const *>(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr, vector::avx2< v256< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store aligned integer values into 256 Bit vector register." );
         _mm256_store_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned float values into 256 Bit vector register." );
         return _mm256_load_ps(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr, vector::avx2< v256< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store aligned float values into 128 Bit vector register." );
         _mm256_store_ps(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }
            
      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned double values into 256 Bit vector register." );
         return _mm256_load_pd(reinterpret_cast< U const * >(p_DataPtr));
      }
      
     template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr, vector::avx2< v256< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store aligned double values into 256 Bit vector register." );
         _mm256_store_pd(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }
   };

   template<typename T, int IOGranularity>
   struct io<avx2<v256<T>>,iov::STREAM, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Stream load integer values into 256 Bit vector register." );
         return _mm256_stream_load_si128(reinterpret_cast<typename avx2< v256< U > >::vector_t const *>(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr, vector::avx2< v256< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Stream store integer values into 256 Bit vector register." );
         _mm256_stream_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return ;
      }
      
   };

   template<typename T, int IOGranularity>
   struct io<avx2<v256<T>>,iov::UNALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 256 Bit vector register." );
         return _mm256_loadu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t const *>(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr, vector::avx2< v256< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned integer values into 256 Bit vector register." );
          _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *> (p_DataPtr),p_vec);
          return;
      }


            
      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned float values into 256 Bit vector register." );
         return _mm256_loadu_ps(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr, vector::avx2< v256< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned float values into 256 Bit vector register." );
         _mm256_storeu_ps(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }
            
      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned double values into 256 Bit vector register." );
         return _mm256_loadu_pd(reinterpret_cast< U const * >(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr, vector::avx2< v256< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned double values into 256 Bit vector register." );
         _mm256_storeu_pd(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }
           
   };

   template<typename T, int IOGranularity>
   struct io<avx2<v256<T>>,iov::UNALIGNEDX, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 256 Bit vector register using lddqu." );
         return _mm256_lddqu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t const *>(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr, vector::avx2< v256< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned integer values into 128 Bit vector register." );
         _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }
      
      
   };


}

#endif //MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_IO_AVX2_H
