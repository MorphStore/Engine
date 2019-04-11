/**
 * @file vector_reg.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_GENERAL_VECTOR_H
#define MORPHSTORE_VECTOR_GENERAL_VECTOR_H

#include <cstdint>

namespace vector {

   template<uint16_t BitWidth, typename T>
   struct vector_view {
      vector_view() = delete;
      using base_t          = T;
      using size_b          = std::integral_constant<uint16_t, BitWidth>;
      using size_B          = std::integral_constant<uint16_t, (BitWidth >> 3)>;
      using alignment       = std::integral_constant<size_t, size_B::value>;
      using element_count   = std::integral_constant<size_t, size_B::value / sizeof(T)>;
   };


   template<typename T>
   using v128 = vector_view<128, T>;
   template<typename T>
   using v256 = vector_view<256, T>;
   template<typename T>
   using v512 = vector_view<512, T>;

}


#endif //MORPHSTORE_VECTOR_GENERAL_VECTOR_H
