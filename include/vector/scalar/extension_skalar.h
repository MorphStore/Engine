/**
 * @file extension_avx2.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_SCALAR_EXTENSION_SCALAR_H
#define MORPHSTORE_VECTOR_SCALAR_EXTENSION_SCALAR_H

#include <cstdint>
#include <type_traits>

#include "vector/general_vector.h"

namespace vector {
    
   template<class VectorReg>
   struct scalar;
   template<typename T>
   struct scalar<v64 <T>> {
      static_assert(std::is_arithmetic<T>::value, "Base type of vector register has to be arithmetic.");
      using vector_helper_t = v64<T>;
      using base_t = typename vector_helper_t::base_t;

      using vector_t = T;
      using size = std::integral_constant<size_t, sizeof(vector_t)>;
      using mask_t = uint16_t;
   };

}

#endif //MORPHSTORE_VECTOR_SCALAR_EXTENSION_SCALAR_H
