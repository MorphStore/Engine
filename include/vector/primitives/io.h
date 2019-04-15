/**
 * @file io.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_PRIMITIVES_IO_H
#define MORPHSTORE_VECTOR_PRIMITIVES_IO_H

#include <vector/general_vector.h>

namespace vector {

   enum class iov {
      ALIGNED,
      UNALIGNED,
      UNALIGNEDX,
      STREAM
   };

   template<class VectorExtension, iov IOVariant, int IOGranularity>
   struct io;
   
   template<class VectorExtension, iov IOVariant, int IOGranularity>
   typename VectorExtension::vector_t
   load(typename VectorExtension::base_t const * const a ) {
       return io<VectorExtension, IOVariant, IOGranularity>::load( a );
   }
   
   template<class VectorExtension, iov IOVariant, int IOGranularity>
   void
   store(typename VectorExtension::base_t * a,  typename VectorExtension::vector_t b ) {
       io<VectorExtension, IOVariant, IOGranularity>::store( a, b );
       return;
   }
   
}
#endif //MORPHSTORE_VECTOR_PRIMITIVES_IO_H
