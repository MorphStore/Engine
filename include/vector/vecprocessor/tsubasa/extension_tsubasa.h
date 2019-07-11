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
#include "immintrin.h"

#include "vector/vector_extension_structs.h"

namespace vector {
   template<class VectorReg>
   struct aurora;

   template<typename T>
   struct aurora< v16k< T > > {
      static_assert(std::is_arithmetic<T>::value, "Base type of vector register has to be arithmetic.");
      static_assert(sizeof(T) == 8, )
      using vector_helper_t = v128<T>;

      using vector_t =
      typename std::conditional<
         std::is_integral<T>::value,    // if T is integer
         __m128i,                       //    vector register = __m128i
         typename std::conditional<
            std::is_same<float, T>::value, // else if T is float
            __m128,                       //    vector register = __m128
            __m128d                       // else [T == double]: vector register = __m128d
         >::type
      >::type;

      using size = std::integral_constant<size_t, sizeof(vector_t)>;
      using mask_t = uint16_t;
   };

}
#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_EXTENSION_TSUBASA_H
