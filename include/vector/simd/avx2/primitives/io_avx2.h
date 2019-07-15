/**
 * @file io.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_IO_AVX2_H
#define MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_IO_AVX2_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/primitives/io.h>

#include <functional>

namespace vectorlib {
    
    
   template<typename T, int IOGranularity>
   struct io<avx2<v256<T>>,iov::ALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned integer values into 256 Bit vector register." );
         return _mm256_load_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t const *>(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, avx2< v256< uint64_t > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store aligned integer values to memory" );
         _mm256_store_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned float values into 256 Bit vector register." );
         return _mm256_load_ps(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, avx2< v256< float > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store aligned float values to memory" );
         _mm256_store_ps(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }
            
      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned double values into 256 Bit vector register." );
         return _mm256_load_pd(reinterpret_cast< U const * >(p_DataPtr));
      }
      
     template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, avx2< v256< double > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store aligned double values to memory" );
         _mm256_store_pd(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }

   };

   template<typename T, int IOGranularity>
   struct io<avx2<v256<T>>,iov::STREAM, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Stream load integer values into 256 Bit vector register." );
         return _mm256_stream_load_si128(reinterpret_cast<typename avx2< v256< U > >::vector_t const *>(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, avx2< v256< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Stream store integer values to memory" );
         _mm256_stream_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return ;
      }
      

      
   };

   template<typename T, int IOGranularity>
   struct io<avx2<v256<T>>,iov::UNALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 256 Bit vector register." );
         return _mm256_loadu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t const *>(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, avx2< v256< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned integer values to memory" );
          _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *> (p_DataPtr),p_vec);
          return;
      }

     
      
      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned float values into 256 Bit vector register." );
         return _mm256_loadu_ps(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, avx2< v256< float > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned float values to memory" );
         _mm256_storeu_ps(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }
            
      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned double values into 256 Bit vector register." );
         return _mm256_loadu_pd(reinterpret_cast< U const * >(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, avx2< v256< double > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned double values to memory" );
         _mm256_storeu_pd(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }
      
    /*! This function compresses the data in a register according to a bitmask (all values with an according set bit will be packed at the beginning of the register), and stores it at a given address.
    * Note: This does not really compress, just shift the values we want to store to the lower bits.
    * If you need a real compress store, copy this code and change the used store-intrinsic to _mm256_maskstore* (and provide the according mask, of course).
    * This function will move to the vector lib someday.
    * @param outPtr The memory address where the vector should be stored
    * @param mask A bitmask with a bit set for every value which is goin to be stored
    * @param vector The 256-bit vector to be comprssed and stored 
    */
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      compressstore( U * p_DataPtr,  avx2< v256< int > > ::vector_t p_vec, int mask ) {
         trace( "[VECTOR] - Store masked unaligned integer values to memory" );
          switch (mask){
          
                    case 0: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr), _mm256_permute4x64_epi64(p_vec,228)); break;
                    case 1: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr), _mm256_permute4x64_epi64(p_vec,228)); break;
                    case 2: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr), _mm256_permute4x64_epi64(p_vec,57)); break;
                    case 3: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr), _mm256_permute4x64_epi64(p_vec,228)); break;
                    case 4: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr), _mm256_permute4x64_epi64(p_vec,78)); break;
                    case 5: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr), _mm256_permute4x64_epi64(p_vec,216)); break;
                    case 6: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr), _mm256_permute4x64_epi64(p_vec,57)); break;
                    case 7: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr), _mm256_permute4x64_epi64(p_vec,228)); break;
                    case 8: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr), _mm256_permute4x64_epi64(p_vec,147)); break;
                    case 9: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr), _mm256_permute4x64_epi64(p_vec,156)); break;
                    case 10: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),_mm256_permute4x64_epi64(p_vec,141)); break;
                    case 11: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),_mm256_permute4x64_epi64(p_vec,180)); break;
                    case 12: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),_mm256_permute4x64_epi64(p_vec,78)); break;
                    case 13: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),_mm256_permute4x64_epi64(p_vec,120)); break;
                    case 14: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),_mm256_permute4x64_epi64(p_vec,57)); break;
                    case 15: _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),_mm256_permute4x64_epi64(p_vec,228)); break;
                }
         
         return ;
      }

      //@todo we should actually provide a specialization (depending on the basetype) here!
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      gather( U const * const p_DataPtr,  avx2< v256< uint64_t > >::vector_t p_vec ) {
         trace( "[VECTOR] - Gather integer values into 256 Bit vector register." );
         return _mm256_i64gather_epi64( reinterpret_cast<const long long int *> (p_DataPtr), p_vec, sizeof(uint64_t));
         
      }
            
           
   };

   template<typename T, int IOGranularity>
   struct io<avx2<v256<T>>,iov::UNALIGNEDX, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename avx2< v256< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 256 Bit vector register using lddqu." );
         return _mm256_lddqu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t const *>(p_DataPtr));
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, avx2< v256< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned integer values to memory" );
         _mm256_storeu_si256(reinterpret_cast<typename avx2< v256< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }
      

      
   };

    //@todo we should actually provide a specialization (depending on the basetype) here!
    template<typename T, int IOGranularity>
    struct io<avx2<v128<T>>,iov::UNALIGNED, IOGranularity> {
     
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename avx2< v128< U > >::vector_t
        gather( U const * const p_DataPtr,  avx2< v128< uint64_t > >::vector_t p_vec ) {
            trace( "[VECTOR] - Gather integer values into 128 Bit vector register." );
            return _mm_i64gather_epi64( reinterpret_cast<const long long int *> (p_DataPtr), p_vec, sizeof(uint64_t));

          }
    };
}

#endif //MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_IO_AVX2_H
