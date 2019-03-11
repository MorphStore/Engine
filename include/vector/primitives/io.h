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
   struct load;

}
#endif //MORPHSTORE_VECTOR_PRIMITIVES_IO_H
