/**
 * @file extension_avx2.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_SIMD_AVX2_EXTENSION_AVX2_H
#define MORPHSTORE_VECTOR_SIMD_AVX2_EXTENSION_AVX2_H

#include <cstdint>
#include <type_traits>
#include "immintrin.h"

#include "vector/general_vector.h"


namespace vector {
   template<class VectorReg>
   struct avx2;

   template<typename T>
   struct avx2< v256< T > > {
      static_assert(std::is_arithmetic<T>::value, "Base type of vector register has to be arithmetic.");
      using vector_helper_t = v256<T>;

      using vector_t =
      typename std::conditional<
         std::is_integral<T>::value,    // if T is integer
         __m256i,                       //    vector register = __m128i
         typename std::conditional<
            std::is_same<float, T>::value, // else if T is float
            __m256,                       //    vector register = __m128
            __m256d                       // else [T == double]: vector register = __m128d
         >::type
      >::type;

      using size = std::integral_constant<size_t, sizeof(vector_t)>;
      using mask_t = uint16_t;
   };

}

#endif //MORPHSTORE_VECTOR_SIMD_AVX2_EXTENSION_AVX2_H
