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


namespace vectorlib {


   template<typename T, int IOGranularity>
   struct io<sse<v128<T>>,iov::ALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned integer values into 128 Bit vector register." );
         return _mm_load_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, sse< v128< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store aligned integer values to memory" );
         _mm_store_si128(reinterpret_cast<typename sse< v128< U > >::vector_t *>(p_DataPtr),p_vec);
         return;
      }



      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned float values into 128 Bit vector register." );
         return _mm_load_ps(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, sse< v128< float > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store aligned float values to memory" );
         _mm_store_ps(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec);
         return;
      }

      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned double values into 128 Bit vector register." );
         return _mm_load_pd(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, sse< v128< double > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store aligned double values to memory" );
         _mm_store_pd(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec);
         return;
      }
   };

   template<typename T, int IOGranularity>
   struct io<sse<v128<T>>,iov::STREAM, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Stream load integer values into 128 Bit vector register." );
         return _mm_stream_load_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, sse< v128< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Stream store integer values to memory" );
         return _mm_stream_si128(reinterpret_cast<typename sse< v128< U > >::vector_t *>(p_DataPtr),p_vec);
      }



   };

   template<typename T, int IOGranularity>
   struct io<sse<v128<T>>,iov::UNALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 128 Bit vector register." );
         return _mm_loadu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, sse< v128< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned integer values to memory" );
         _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec);
         return;
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned float values into 128 Bit vector register." );
         return _mm_loadu_ps(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, sse< v128< float > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned float values to memory" );
         _mm_storeu_ps(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec);
         return;
      }

      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned double values into 128 Bit vector register." );
         return _mm_loadu_pd(reinterpret_cast< U const * >(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, sse< v128< double > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned double values to memory" );
         _mm_storeu_pd(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec);
         return;
      }

   };


   template<typename T>
   struct io<sse<v128<T>>,iov::UNALIGNED, 64> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      compressstore( U * p_DataPtr,  typename sse< v128< U > >::vector_t p_vec, int mask ) {
         trace( "[VECTOR] - Store masked unaligned integer values to memory" );

         switch (mask){
            case 0:    return; //store nothing
            case 1:    _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec);
                        return; //store everything
            case 2:    p_vec=_mm_shuffle_epi8(p_vec, _mm_set_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8)); //move upper 64 bit to beginning of register and store it to memory
                        _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec);
                        return;
            case 3:    _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec); //store everything
                        return;
         }

         return ;
      }
   };

   template<typename T>
   struct io<sse<v128<T>>,iov::UNALIGNED, 32> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      compressstore( U * p_DataPtr,  typename sse< v128< U > >::vector_t p_vec, int mask ) {
         trace( "[VECTOR] - Store masked unaligned integer values to memory" );

         switch (mask){
            case 0b0000: return;
            case 0b0001: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec); return;                   //Store everything
            case 0b0010: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),_mm_srli_si128(p_vec,4)); return; //Bytewise shift
            case 0b0011: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec); return;                   //Store everything
            case 0b0100: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),_mm_srli_si128(p_vec,8)); return;
            case 0b0101: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),_mm_shuffle_epi8(p_vec,_mm_set_epi8(15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0))); return;   //Shuffel Bytewise and Store
            case 0b0110: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),_mm_srli_si128(p_vec,4)); return;
            case 0b0111: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec); return;
            case 0b1000: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),_mm_srli_si128(p_vec,12)); return;
            case 0b1001: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),_mm_shuffle_epi8(p_vec,_mm_set_epi8(11, 10, 9, 8, 7, 6, 5, 4, 15, 14, 13, 12, 3, 2, 1, 0))); return;
            case 0b1010: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),_mm_shuffle_epi8(p_vec,_mm_set_epi8(11, 10, 9, 8, 3, 2, 1, 0, 15, 14, 13, 12, 7, 6, 5, 4))); return;
            case 0b1011: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),_mm_shuffle_epi8(p_vec,_mm_set_epi8(11, 10, 9, 8, 15, 14, 13, 12, 7, 6, 5, 4, 3, 2, 1, 0))); return;
            case 0b1100: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),_mm_srli_si128(p_vec,8)); return;
            case 0b1101: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),_mm_shuffle_epi8(p_vec,_mm_set_epi8(7, 6, 5, 4, 15, 14, 13, 12, 11, 10, 9, 8, 3, 2, 1, 0))); return;
            case 0b1110: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),_mm_srli_si128(p_vec,4)); return;
            case 0b1111: _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec); return;
         }
         return ;
      }
   };
//probably has the same error as io_avx2 has. so this will be last.
   template<typename T>
   struct io<sse<v128<T>>,iov::UNALIGNED, 16> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      compressstore( U * p_DataPtr,  typename sse< v128< U > >::vector_t p_vec, int mask ) {
         trace( "[VECTOR] - Store masked unaligned integer values to memory" );

         int8_t matched = 0;
         while(mask != 0){
            if((mask & 0x1) == 0x1){
               if(matched == 0){
                  _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec);
               }
               p_DataPtr ++;
            }else{
               matched = 0;
            }
            mask = (mask >> 1) & 0x7FFF;
            p_vec = _mm_srli_si128(p_vec,1);
         }
         return ;
      }
   };

   template<typename T>
   struct io<sse<v128<T>>,iov::UNALIGNED, 8> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      compressstore( U * p_DataPtr,  typename sse< v128< U > >::vector_t p_vec, int mask ) {
         trace( "[VECTOR] - Store masked unaligned integer values to memory" );
         int8_t matched = 0;
         while(mask != 0){
            if((mask & 0x1) == 0x1){
               if(matched == 0){
                  _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec);
               }
               p_DataPtr ++;
            }else{
               matched = 0;
            }
            mask = (mask >> 1) & 0x7FFF;
            p_vec = _mm_srli_si128(p_vec,2);
         }
         return ;
      }
   };

   template<typename T, int IOGranularity, int Scale>
   struct gather_t<sse<v128<T>>, IOGranularity, Scale> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      apply( U const * const p_DataPtr,  sse< v128< uint64_t > >::vector_t p_vec ) {
         trace( "[VECTOR] - Gather integer values into 128 Bit vector register." );
         return _mm_set_epi64x(
               *reinterpret_cast<uint64_t const *>(reinterpret_cast<uint8_t const *>(p_DataPtr) + _mm_extract_epi64(p_vec,1) * Scale),
               *reinterpret_cast<uint64_t const *>(reinterpret_cast<uint8_t const *>(p_DataPtr) + _mm_extract_epi64(p_vec,0) * Scale)
         );
        // return _mm256_i64gather_epi64( reinterpret_cast<typename avx2< v256< int > >::vector_t const *> (p_DataPtr), p_vec, sizeof(uint64_t));

      }

   };

   template<typename T, int IOGranularity>
   struct io<sse<v128<T>>,iov::UNALIGNEDX, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename sse< v128< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned integer values into 128 Bit vector register using lddqu." );
         return _mm_lddqu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
      }

      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, sse< v128< int > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store unaligned integer values to memory" );
         _mm_storeu_si128(reinterpret_cast<typename sse< v128< U > >::vector_t  *>(p_DataPtr),p_vec);
         return;
      }



   };

}


#endif //MORPHSTORE_VECTOR_SIMD_SSE_PRIMITIVES_IO_SSE_H
