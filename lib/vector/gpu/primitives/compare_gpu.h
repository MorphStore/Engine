#ifndef MORPHSTORE_VECTOR_GPU_PRIMITIVES_COMPARE_GPU_H
#define MORPHSTORE_VECTOR_GPU_PRIMITIVES_COMPARE_GPU_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include "../extension_gpu.h"
#include "../../primitives/compare.h"
namespace vectorlib{

    __forceinline__ __device__ bool check_equal_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
        return (p_vec1_value == p_vec2_value);
    }

    __global__
    void check_equal(uint64_t* p_vec1, uint64_t* p_vec2, gpu<v2048<uint64_t>>::mask_t* mask) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        atomicAdd(mask,check_equal_on_device(p_vec1[index], p_vec2[index]) << index);
    }

    template<>
    struct equal<gpu<v2048<uint64_t>>/*, 64*/> {
    	MSV_CXX_ATTRIBUTE_FORCE_INLINE
    	static typename gpu<v2048<uint64_t>>::mask_t*
    	apply(
        	typename gpu<v2048<uint64_t>>::vector_t const p_vec1,
        	typename gpu<v2048<uint64_t>>::vector_t const p_vec2
      	) {
            gpu<v2048<uint64_t>>::mask_t* mask;
            cudaMalloc((void**)&mask, sizeof(uint32_t));
            check_equal<<<1,32>>>(p_vec1, p_vec2, mask);
            return mask;
        }
    };

    __forceinline__ __device__ bool check_less_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
        return (p_vec1_value < p_vec2_value);
    }

    __global__
    void check_less(uint64_t* p_vec1, uint64_t* p_vec2, gpu<v2048<uint64_t>>::mask_t* mask) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        atomicAdd(mask,check_less_on_device(p_vec1[index], p_vec2[index]) << index);
    }

    template<>
    struct less<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename gpu<v2048<uint64_t>>::mask_t*
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const p_vec1,
            typename gpu<v2048<uint64_t>>::vector_t const p_vec2
        ) {
            gpu<v2048<uint64_t>>::mask_t* mask;
            cudaMalloc((void**)&mask, sizeof(uint32_t));
            check_less<<<1,32>>>(p_vec1, p_vec2, mask);
            return mask;
        }
    };

    __forceinline__ __device__ bool check_lessequal_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
        return (p_vec1_value <= p_vec2_value);
    }

    __global__
    void check_lessequal(uint64_t* p_vec1, uint64_t* p_vec2, gpu<v2048<uint64_t>>::mask_t* mask) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        atomicAdd(mask,check_lessequal_on_device(p_vec1[index], p_vec2[index]) << index);
    }

    template<>
    struct lessequal<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename gpu<v2048<uint64_t>>::mask_t*
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const p_vec1,
            typename gpu<v2048<uint64_t>>::vector_t const p_vec2
        ) {
            gpu<v2048<uint64_t>>::mask_t* mask;
            cudaMalloc((void**)&mask, sizeof(uint32_t));
            check_lessequal<<<1,32>>>(p_vec1, p_vec2, mask);
            return mask;
        }
    };

    __forceinline__ __device__ bool check_greater_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
        return (p_vec1_value > p_vec2_value);
    }

    __global__
    void check_greater(uint64_t* p_vec1, uint64_t* p_vec2, gpu<v2048<uint64_t>>::mask_t* mask) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        atomicAdd(mask,check_greater_on_device(p_vec1[index], p_vec2[index]) << index);
    }

    template<>
    struct greater<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename gpu<v2048<uint64_t>>::mask_t*
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const p_vec1,
            typename gpu<v2048<uint64_t>>::vector_t const p_vec2
        ) {
            gpu<v2048<uint64_t>>::mask_t* mask;
            cudaMalloc((void**)&mask, sizeof(uint32_t));
            check_greater<<<1,32>>>(p_vec1, p_vec2, mask);
            return mask;
        }
    };

    __forceinline__ __device__ bool check_greaterequal_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
        return (p_vec1_value >= p_vec2_value);
    }

    __global__
    void check_greaterequal(uint64_t* p_vec1, uint64_t* p_vec2, gpu<v2048<uint64_t>>::mask_t* mask) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        atomicAdd(mask,check_greaterequal_on_device(p_vec1[index], p_vec2[index]) << index);
    }

    template<>
    struct greaterequal<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename gpu<v2048<uint64_t>>::mask_t*
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const p_vec1,
            typename gpu<v2048<uint64_t>>::vector_t const p_vec2
        ) {
            gpu<v2048<uint64_t>>::mask_t* mask;
            cudaMalloc((void**)&mask, sizeof(uint32_t));
            check_greaterequal<<<1,32>>>(p_vec1, p_vec2, mask);
            return mask;
        }
    };

    __global__
    void count_matches_device(gpu<v2048<uint64_t>>::mask_t* p_mask, uint8_t* number_matches) {
        *number_matches = __popc(*p_mask);
    }

    template<>
    struct count_matches<gpu<v2048<uint64_t>>> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static uint8_t*
        apply(
            typename gpu<v2048<uint64_t>>::mask_t* const p_mask
        ) {
            uint8_t* number_matches;
            cudaMalloc((void**)&number_matches, sizeof(uint8_t));
            count_matches_device<<<1,1>>>(p_mask, number_matches);
            return number_matches;
      }
   };
}

#endif
