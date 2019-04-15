/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   io_avx512.h
 * Author: Annett
 *
 * Created on 12. April 2019, 12:21
 */

#ifndef IO_AVX512_H
#define IO_AVX512_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx512/extension_avx512.h>
#include <vector/primitives/io.h>

#include <functional>

namespace vector {
  
   template<typename T, int IOGranularity>
   struct io<avx512<v512<T>>,iov::ALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx512< v512< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned integer values into 512 Bit vector register." );
         return _mm512_load_si512(/*reinterpret_cast<typename avx512< v512< U > >::vector_t const *>*/( void * )(p_DataPtr));
      }
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr, avx512< v512< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Loading aligned integer values to memory" );
         _mm512_store_si512(/*reinterpret_cast<typename avx512< v512< U > >::vector_t const *>*/( void * )(p_DataPtr),p_vec);
         return;
      }
      
      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx512< v512< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned float values into 512 Bit vector register." );
         return _mm512_load_ps(reinterpret_cast< U const * >(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr , avx512< v512< float > >::vector_t p_vec ) {
         trace( "[VECTOR] - Loading aligned float values  to memory" );
         _mm512_store_ps(( void * )(p_DataPtr),p_vec);
         return;
      }

      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx512< v512< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned double values into 512 Bit vector register." );
         return _mm512_load_pd(reinterpret_cast< U const * >(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr,  avx512< v512< double > >::vector_t p_vec ) {
         trace( "[VECTOR] - Loading aligned double values  to memory" );
         _mm512_store_pd(( void * )(p_DataPtr), p_vec);
         return;
      }
      
    
   };

   template<typename T, int IOGranularity>
   struct io<avx512<v512<T>>,iov::STREAM, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx512< v512< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Stream load integer values into 512 Bit vector register." );
         return _mm512_stream_load_si512(reinterpret_cast<typename avx512< v512< U > >::vector_t const *>(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U const * const p_DataPtr,  avx512< v512< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Stream store integer values  to memory" );
         _mm512_stream_si512((void *)p_DataPtr, p_vec);
         return ;
      }
      
   
   };

   template<typename T, int IOGranularity>
   struct io<avx512<v512<T>>,iov::UNALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx512< v512< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 512 Bit vector register." );
         return _mm512_loadu_si512(reinterpret_cast<typename avx512< v512< U > >::vector_t const *>(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr,  avx512< v512< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned integer values  to memory" );
         _mm512_storeu_si512((void *)p_DataPtr, p_vec);
         return ;
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      compressstore( U * p_DataPtr,  avx512< v512< int > >::vector_t p_vec, int mask ) {
         trace( "[VECTOR] - Store masked unaligned integer values to memory" );
         _mm512_mask_compressstoreu_epi64((void *)p_DataPtr, mask,p_vec);
         return ;
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx512< v512< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned float values into 512 Bit vector register." );
         return _mm_loadu_ps(reinterpret_cast< U const * >(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr,  avx512< v512< float > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned floating values to memory" );
         _mm512_storeu_ps((void *)p_DataPtr, p_vec);
         return ;
      }

      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx512< v512< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned double values into 512 Bit vector register." );
         return _mm_loadu_pd(reinterpret_cast< U const * >(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr, avx512< v512< double > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned double values  to memory" );
         _mm512_storeu_pd((void *)p_DataPtr, p_vec);
         return ;
      }
   };

   template<typename T, int IOGranularity>
   struct io<avx512<v512<T>>,iov::UNALIGNEDX, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static typename avx512< v512< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 512 Bit vector register using lddqu." );
         return _mm512_loadu_si512(reinterpret_cast<typename avx512< v512< U > >::vector_t const *>(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U * p_DataPtr,  avx512< v512< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned integer values  to memory" );
         _mm512_storeu_si512((void *)p_DataPtr, p_vec);
         return ;
      }
      
 
   };
   
    template<typename T, int IOGranularity>
    struct io<avx512<v256<T>>,iov::UNALIGNED, IOGranularity> {
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      compressstore( U const * const p_DataPtr,  avx512< v256< int > > ::vector_t p_vec, int mask ) {
         trace( "[VECTOR] - Store masked unaligned integer values to memory" );
         _mm256_mask_compressstoreu_epi64((void *)p_DataPtr,mask, p_vec);
         return ;
      }
    };
    
    template<typename T, int IOGranularity>
    struct io<avx512<v128<T>>,iov::UNALIGNED, IOGranularity> {
    template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      compressstore( U const * const p_DataPtr,  avx512< v128< int > >::vector_t p_vec, int mask ) {
         trace( "[VECTOR] - Store masked unaligned integer values to memory" );
         _mm_mask_compressstoreu_epi64((void *)p_DataPtr,mask, p_vec);
         return ;
      }
    };
  

}

#endif /* IO_AVX512_H */

