/**
 * @file vector_reg.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_GENERAL_VECTOR_H
#define MORPHSTORE_VECTOR_GENERAL_VECTOR_H

#include <cstdint>
#include <type_traits>
#include <cstddef>
#include <core/utils/preprocessor.h>
namespace vector {

   template<uint16_t BitWidth, typename T>
   struct vector_view {
      vector_view() = delete;
      using base_t          = T;
      using size_bit        = std::integral_constant<uint16_t, BitWidth>;
      using size_byte       = std::integral_constant<uint16_t, (BitWidth>>3) >;
      using alignment       = std::integral_constant<size_t, size_byte::value>;
      using element_count   = std::integral_constant<size_t, size_byte::value / sizeof(T)>;
      using granularity     = std::integral_constant<size_t, sizeof(T)<<3>;
   };

   template<typename T>
   //using v1 = vector_view<(sizeof(T)<<3), T>;
   using v64 = vector_view<64, T>;
   
   template<typename T>
   using v128 = vector_view<128, T>;
   template<typename T>
   using v256 = vector_view<256, T>;
   template<typename T>
   using v512 = vector_view<512, T>;

#define IMPORT_VECTOR_BOILER_PLATE(VectorExtension) \
   using vector_element_count MSV_CXX_ATTRIBUTE_PPUNUSED = typename VectorExtension::vector_helper_t::element_count; \
   using base_t MSV_CXX_ATTRIBUTE_PPUNUSED = typename VectorExtension::vector_helper_t::base_t; \
   using vector_size_bit MSV_CXX_ATTRIBUTE_PPUNUSED = typename VectorExtension::vector_helper_t::size_bit; \
   using vector_size_byte MSV_CXX_ATTRIBUTE_PPUNUSED = typename VectorExtension::vector_helper_t::size_byte; \
   using vector_alignment MSV_CXX_ATTRIBUTE_PPUNUSED = typename VectorExtension::vector_helper_t::alignment; \
   using vector_t MSV_CXX_ATTRIBUTE_PPUNUSED = typename VectorExtension::vector_t; \
   using vector_size MSV_CXX_ATTRIBUTE_PPUNUSED = typename VectorExtension::size; \
   using vector_mask_t MSV_CXX_ATTRIBUTE_PPUNUSED = typename VectorExtension::mask_t; \
   using vector_base_t_granularity MSV_CXX_ATTRIBUTE_PPUNUSED = typename VectorExtension::vector_helper_t::granularity;


}


#endif //MORPHSTORE_VECTOR_GENERAL_VECTOR_H
