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
      using size_bit        = std::integral_constant<uint16_t, BitWidth>;
      using size_byte       = std::integral_constant<uint16_t, (BitWidth >
      using alignment       = std::integral_constant<size_t, size_B::value>;
      using element_count   = std::integral_constant<size_t, size_B::value / sizeof(T)>;
   };


   template<typename T>
   using v128 = vector_view<128, T>;
   template<typename T>
   using v256 = vector_view<256, T>;
   template<typename T>
   using v512 = vector_view<512, T>;


#define IMPORT_VECTOR_BOILER_PLATE(VectorExtension) using vector_element_count = VectorExtension::vector_helper_t::element_count; \
   using vector_base_t = typename VectorExtension::vector_helper_t::base_t; \
   using vector_size_bit = VectorExtension::vector_helper_t::size_bit; \
   using vector_size_byte = VectorExtension::vector_helper_t::size_byte; \
   using vector_alignment = VectorExtension::vector_helper_t::alignment; \
   using vector_element_count = VectorExtension::vector_helper_t::element_count; \
   using vector_t = VectorExtension::vector_t; \
   using vector_size = VectorExtension::size; \
   using vector_mask_t = VectorExtension::mask_t; \
   using size = std::integral_constant<size_t, sizeof(vector_t)>; \
   using mask_t = uint16_t;


}


#endif //MORPHSTORE_VECTOR_GENERAL_VECTOR_H
