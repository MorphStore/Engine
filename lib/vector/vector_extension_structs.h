/**
 * @file vector_reg.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */
#ifndef MORPHSTORE_VECTOR_VECTOR_EXTENSION_STRUCTS_H
#define MORPHSTORE_VECTOR_VECTOR_EXTENSION_STRUCTS_H

#include "general_vector_extension.h"

#  ifdef AVX512
#    include <vector/simd/avx512/extension_avx512.h>
#  endif

#  ifdef AVXTWO
#    include <vector/simd/avx2/extension_avx2.h>
#  endif

#  ifdef SSE
#    include <vector/simd/sse/extension_sse.h>
#  endif

#ifdef NEON
#   include <vector/simd/neon/extension_neon.h>
#endif

#ifdef __CUDACC__
#   include <vector/gpu/extension_gpu.h>
#endif

#include "scalar/extension_scalar.h"
#include "vv/extension_virtual_vector.h"

namespace vectorlib{

#ifdef __CUDACC__
   #define IMPORT_VECTOR_BOILER_PLATE(VectorExtension) \
      using vector_element_count = typename VectorExtension::vector_helper_t::element_count; \
      using base_t = typename VectorExtension::vector_helper_t::base_t; \
      using vector_size_bit = typename VectorExtension::vector_helper_t::size_bit; \
      using vector_size_byte = typename VectorExtension::vector_helper_t::size_byte; \
      using vector_alignment = typename VectorExtension::vector_helper_t::alignment; \
      using vector_t = typename VectorExtension::vector_t; \
      using vector_size = typename VectorExtension::size; \
      using vector_mask_t = typename VectorExtension::mask_t; \
      using vector_base_t_granularity = typename VectorExtension::vector_helper_t::granularity;

#else
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
#endif

}
#endif //MORPHSTORE_VECTOR_VECTOR_EXTENSION_STRUCTS_H
