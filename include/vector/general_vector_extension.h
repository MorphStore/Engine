/**
 * @file vector_reg.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_GENERAL_VECTOR_EXTENSION_H
#define MORPHSTORE_VECTOR_GENERAL_VECTOR_EXTENSION_H

#include <cstdint>
#include <type_traits>
#include <cstddef>
#include <core/utils/preprocessor.h>
namespace vectorlib {

   template<uint16_t BitWidth, typename T>
   struct vector_view {
      vector_view() = delete;

      using base_t          = T;
      using size_bit        = std::integral_constant<uint16_t, BitWidth>;
      using size_byte       = std::integral_constant<uint16_t, (BitWidth >> 3)>;
      using alignment       = std::integral_constant<size_t, size_byte::value>;
      using element_count   = std::integral_constant<size_t, size_byte::value / sizeof(T)>;
      using granularity     = std::integral_constant<size_t, sizeof(T) << 3>;
   };

   template<typename T>
   //using v1 = vector_view<(sizeof(T)<<3), T>;
   using v64 = vector_view<64, T>;
   template<typename T>
   using v32 = vector_view<32, T>;
   template<typename T>
   using v128 = vector_view<128, T>;
   template<typename T>
   using v256 = vector_view<256, T>;
   template<typename T>
   using v512 = vector_view<512, T>;
   template<typename T>
   using v1024 = vector_view<1024, T>;
   template<typename T>
   using v2048 = vector_view<2048, T>;
   template<typename T>
   using v4096 = vector_view<4096, T>;
   template<typename T>
   using v8192 = vector_view<8192, T>;
   template<typename T>
   using v16384 = vector_view<16384, T>;

}

#endif //MORPHSTORE_VECTOR_GENERAL_VECTOR_EXTENSION_H
