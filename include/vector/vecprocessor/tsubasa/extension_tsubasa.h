/**
 * @file extension_tsubasa.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_EXTENSION_TSUBASA_H
#define MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_EXTENSION_TSUBASA_H

#include <cstdint>
#include <type_traits>
#include "veintrin.h"

#include "vector/vector_extension_structs.h"

namespace vectorlib {
   template<class VectorReg>
   struct aurora;

   template<typename T>
   struct aurora< v16k< T > > {
      static_assert(std::is_arithmetic<T>::value, "Base type of vector register has to be arithmetic.");
      static_assert(sizeof(T) == 8, "Only base type with a size of 8 Byte are supported right now." );
      using vector_helper_t = v16k<T>;
      using base_t = typename vector_helper_t::base_t;

      using vector_t = __vr;

      using size = std::integral_constant<size_t, sizeof(vector_t)>;
      using mask_t = __vm256; //__vm512
   };

}
#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_EXTENSION_TSUBASA_H
