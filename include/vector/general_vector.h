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
   struct vector_reg {
      static constexpr uint16_t size_b = BitWidth / 8;
      static constexpr uint16_t element_count = size_b / sizeof(T);
   };


   template<typename T>
   using v128 = vector_reg<128, T>;
   template<typename T>
   using v256 = vector_reg<256, T>;
   template<typename T>
   using v512 = vector_reg<512, T>;

}


#endif //MORPHSTORE_VECTOR_GENERAL_VECTOR_H
