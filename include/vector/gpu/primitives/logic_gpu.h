#ifndef MORPHSTORE_VECTOR_GPU_PRIMITIVES_LOGIC_GPU_H
#define MORPHSTORE_VECTOR_GPU_PRIMITIVES_LOGIC_GPU_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/gpu/extension_gpu.h>
#include <vector/primitives/logic.h>
namespace vectorlib{

    __forceinline__ __device__ uint64_t bitwise_or_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
        return p_vec1_value | p_vec2_value;
    }

    __global__
    void bitwise_or_elements(uint64_t* p_In1, uint64_t* p_In2) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_In1[index] = bitwise_or_on_device(p_In1[index], p_In2[index]);
    }

    __global__
    void bitwise_and_masks(gpu<v2048<uint64_t>>::mask_t* p_In1, gpu<v2048<uint64_t>>::mask_t* p_In2, gpu<v2048<uint64_t>>::mask_t* mask_anded) {
        *mask_anded = *p_In1 & *p_In2;
    }

    __forceinline__ __device__ uint64_t bitwise_and_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
        return p_vec1_value & p_vec2_value;
    }

    __global__
    void bitwise_and_elements(uint64_t* p_In1, uint64_t* p_In2) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_In1[index] = bitwise_and_on_device(p_In1[index], p_In2[index]);
    }

   template<typename T>
   //struct logic<gpu<v2048<T>>, gpu<v2048<T>>::vector_helper_t::size_bit::value > {
   struct logic<gpu<v2048<T>>,64 > {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename gpu<v2048<T>>::vector_t
      bitwise_and( typename gpu<v2048<T>>::vector_t const & p_In1, typename gpu<v2048<T>>::vector_t const & p_In2) {
            bitwise_and_elements<<<1,32>>>(p_In1, p_In2);
            return p_In1;
      }
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename gpu<v2048<T>>::mask_t*
      bitwise_and( typename gpu<v2048<T>>::mask_t* const p_In1, typename gpu<v2048<T>>::mask_t* const p_In2) {
      //bitwise_and( uint32_t* p_In1, uint32_t* p_In2) {
          gpu<v2048<uint64_t>>::mask_t* mask_anded;
          cudaMalloc((void**)&mask_anded, sizeof(uint32_t));
          bitwise_and_masks<<<1,1>>>(p_In1, p_In2, mask_anded);
          return mask_anded;
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename gpu<v2048<T>>::vector_t
      bitwise_or( typename gpu<v2048<T>>::vector_t const & p_In1, typename gpu<v2048<T>>::vector_t const & p_In2) {
            bitwise_or_elements<<<1,32>>>(p_In1, p_In2);
            return p_In1;
      }
   };
}
#endif