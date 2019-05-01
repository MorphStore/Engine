/**
 * @file vector_reg.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_GENERAL_VECTOR_H
#define MORPHSTORE_VECTOR_GENERAL_VECTOR_H

#include <cstdint>
#include <vector/preprocessor.h>
namespace vector {

   template<uint16_t BitWidth, typename T>
   struct vector_view {
      vector_view() = delete;
      using base_t          = T;
      using base_type_size_bit = std::integral_constant<uint16_t, sizeof(T)<<3>;
      using size_bit        = std::integral_constant<uint16_t, BitWidth>;
      using size_byte       = std::integral_constant<uint16_t, (BitWidth>>3) >;
      using alignment       = std::integral_constant<size_t, size_byte::value>;
      using element_count   = std::integral_constant<size_t, size_byte::value / sizeof(T)>;
      using granularity     = std::integral_constant<size_t, sizeof(T)*8>;
   };


   template<typename T>
   using v128 = vector_view<128, T>;
   template<typename T>
   using v256 = vector_view<256, T>;
   template<typename T>
   using v512 = vector_view<512, T>;


#define IMPORT_VECTOR_BOILER_PLATE(VectorExtension) \
   using vector_element_count = typename VectorExtension::vector_helper_t::element_count; \
   using vector_base_t = typename VectorExtension::vector_helper_t::base_t; \
   using vector_base_type_size_bit = typename VectorExtension::vector_helper_t::base_type_size_bit; \
   using vector_size_bit = typename VectorExtension::vector_helper_t::size_bit; \
   using vector_size_byte = typename VectorExtension::vector_helper_t::size_byte; \
   using vector_alignment = typename VectorExtension::vector_helper_t::alignment; \
   using vector_element_count = typename VectorExtension::vector_helper_t::element_count; \
   using vector_t = typename VectorExtension::vector_t; \
   using vector_size = typename VectorExtension::size; \
   using vector_mask_t = typename VectorExtension::mask_t; \
   using vector_base_t_granularity = typename VectorExtension::vector_helper_t::granularity;


}


#endif //MORPHSTORE_VECTOR_GENERAL_VECTOR_H
