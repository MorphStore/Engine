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
#include <vector/vecprocessor/tsubasa/extension_tsubasa.h>
#include <vector/primitives/io.h>


namespace vector {

   template< typename T, int IOGranularity >
   struct io< aurora< v16k< T > >, iov::ALIGNED, IOGranularity > {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename aurora< v16k< U > >::vector_t
      load( T const * const p_DataPtr ) {
         trace( "[VECTOR] - \"Stream\" Load aligned integer values into 16k Bit Vector register. It's a normal load." );
         return _ve_vld_vss( sizeof( T ), p_DataPtr );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( T *  p_DataPtr, typename aurora< v16k< T > >::vector_t /*const &*/ p_Vec ) {
         trace( "[VECTOR] - Store aligned integer values to memory" );
         _ve_vst_vss( p_Vec, sizeof( T ), p_DataPtr );
         return;
      }
   };
   template< typename T, int IOGranularity >
   struct io< aurora< v16k< T > >, iov::UNALIGNED, IOGranularity > {
      // Unaligned load not supported

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( T *  p_DataPtr, typename aurora< v16k< T > >::vector_t /*const &*/ p_Vec ) {
         trace( "[VECTOR] - Store aligned integer values to memory" );
         _ve_vst_vss( p_Vec, sizeof( T ), p_DataPtr );
         return;
      }

      //uint64_t has to be a _ve_vbrd_vs_i64
      template< typename U = T, typename std::enable_if< sizeof( U ) == 8, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      compressstore(
         T * p_DataPtr,
         typename aurora< v16k< T > >::vector_t /*const &*/ p_Vec,
         typename aurora< v16k< T > >::mask_t /*const &*/ p_Mask ) {
         trace( "[VECTOR] - Store active lanes from a vector register in a consecutive manner (compressed)." );
         _ve_vst_vss(
            _ve_vcp_vvmv( p_Vec, p_Mask, _ve_vbrd_vs_i64( 0 ) ),
            sizeof( T ),
            p_DataPtr
         );
      }

      //uint64_t has to be a 3, uint32_t a 2
      template< typename U = T, typename std::enable_if< sizeof( U ) == 8, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename aurora< v16k< T > >::vector_t
      gather(
         T const * const p_DataPtr,
         typename aurora< v16k< T > >::vector_t /*const &*/ p_Vec
      ) {
         return _ve_vgt_vv( _ve_vsfa_vvss( p_Vec, 3, ( unsigned long int ) p_DataPtr ) );
      }

      //uint64_t has to be a 3, uint32_t a 2
      template< typename U = T, typename std::enable_if< sizeof( U ) == 8, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      scatter(
         T * p_DataPtr,
         typename aurora< v16k< T > >::vector_t /*const &*/ p_Vec,
         typename aurora< v16k< T > >::vector_t /*const &*/ p_Idx
      ) {
         _ve_vsc_vv( p_Vec, _ve_vsfa_vvss( p_Idx, 3, ( unsigned long int ) p_DataPtr ) );
      }
   };

}

#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_IO_TSUBASA_H
