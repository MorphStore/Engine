#ifndef MORPHSTORE_VECTOR_GPU_PRIMITIVES_CALC_GPU_H
#define MORPHSTORE_VECTOR_GPU_PRIMITIVES_CALC_GPU_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/gpu/extension_gpu.h>
#include <vector/primitives/calc.h>

namespace vectorlib{

    // __forceinline__ __device__ uint64_t add_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
    //     return p_vec1_value + p_vec2_value;
    // }

    // __global__
    // void add_elements(uint64_t* p_vec1, uint64_t* p_vec2) {
    //     uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
    //     p_vec1[index] = add_on_device(p_vec1[index], p_vec2[index]);
    // }

    __global__
    void add_elements(uint64_t* p_vec1, uint64_t* p_vec2) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec1[index] = p_vec1[index] + p_vec2[index];
    }

    template<>
    struct add<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
        typename gpu<v2048<uint64_t>>::vector_t
        apply(
        	typename gpu<v2048<uint64_t>>::vector_t const & p_vec1,
        	typename gpu<v2048<uint64_t>>::vector_t const & p_vec2//,
            //int const & N
        ){
            //add_elements<<<1*N,32>>>(p_vec1, p_vec2);
            add_elements<<<1,32>>>(p_vec1, p_vec2);
            return p_vec1;
    	}
    };

    __forceinline__ __device__ uint64_t sub_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
        return p_vec1_value - p_vec2_value;
    }

    __global__
    void sub_elements(uint64_t* p_vec1, uint64_t* p_vec2) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec1[index] = sub_on_device(p_vec1[index], p_vec2[index]);
    }

    template<>
    struct sub<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
        typename gpu<v2048<uint64_t>>::vector_t
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec1,
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec2
        ){
            sub_elements<<<1,32>>>(p_vec1, p_vec2);
            return p_vec1;
        }
    };

    __forceinline__ __device__ uint64_t min_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
        return (p_vec1_value < p_vec2_value) ? p_vec1_value : p_vec2_value;
    }

    __global__
    void min_elements(uint64_t* p_vec1, uint64_t* p_vec2) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec1[index] = min_on_device(p_vec1[index], p_vec2[index]);
    }

    template<>
    struct min<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
        typename gpu<v2048<uint64_t>>::vector_t
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec1,
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec2
        ){
            min_elements<<<1,32>>>(p_vec1, p_vec2);
            return p_vec1;
        }
    };

    __forceinline__ __device__ uint64_t hadd_on_device(uint64_t p_vec1_value) {
        for (int offset = 16; offset > 0; offset /= 2)
            p_vec1_value += __shfl_down_sync(0xffffffff, p_vec1_value, offset);
        return p_vec1_value;
    }

    __global__
    void hadd_elements(uint64_t* p_vec1) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec1[index] = hadd_on_device(p_vec1[index]);
    }

    template<>
    struct hadd<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
        typename gpu<v2048<uint64_t>>::base_t*
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec1
        ){
            hadd_elements<<<1,32>>>(p_vec1);
            return &p_vec1[0];
        }
    };

    __forceinline__ __device__ uint64_t mul_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
        return p_vec1_value * p_vec2_value;
    }

    __global__
    void mul_elements(uint64_t* p_vec1, uint64_t* p_vec2) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec1[index] = mul_on_device(p_vec1[index], p_vec2[index]);
    }

    template<>
    struct mul<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
        typename gpu<v2048<uint64_t>>::vector_t
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec1,
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec2
        ){
            mul_elements<<<1,32>>>(p_vec1, p_vec2);
            return p_vec1;
        }
    };

    __forceinline__ __device__ uint64_t div_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
        return p_vec1_value / p_vec2_value;
    }

    __global__
    void div_elements(uint64_t* p_vec1, uint64_t* p_vec2) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec1[index] = div_on_device(p_vec1[index], p_vec2[index]);
    }

    template<>
    struct div<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
        typename gpu<v2048<uint64_t>>::vector_t
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec1,
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec2
        ){
            div_elements<<<1,32>>>(p_vec1, p_vec2);
            return p_vec1;
        }
    };

    __forceinline__ __device__ uint64_t mod_on_device(uint64_t p_vec1_value, uint64_t p_vec2_value) {
        return p_vec1_value % p_vec2_value;
    }

    __global__
    void mod_elements(uint64_t* p_vec1, uint64_t* p_vec2) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec1[index] = div_on_device(p_vec1[index], p_vec2[index]);
    }

    template<>
    struct mod<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
        typename gpu<v2048<uint64_t>>::vector_t
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec1,
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec2
        ){
            mod_elements<<<1,32>>>(p_vec1, p_vec2);
            return p_vec1;
        }
    };

    __forceinline__ __device__ uint64_t inv_on_device(uint64_t p_vec1_value) {
        return -p_vec1_value;
    }

    __global__
    void inv_elements(uint64_t* p_vec1) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec1[index] = inv_on_device(p_vec1[index]);
    }

    template<>
    struct inv<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
        typename gpu<v2048<uint64_t>>::vector_t
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec1
        ){
            inv_elements<<<1,32>>>(p_vec1);
            return p_vec1;
        }
    };

    __forceinline__ __device__ uint64_t shift_left_on_device(uint64_t p_vec1_value, int p_distance) {
        return p_vec1_value << p_distance;
    }

    __global__
    void shift_left_elements(uint64_t* p_vec1, int p_distance) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec1[index] = shift_left_on_device(p_vec1[index], p_distance);
    }

    template<>
    struct shift_left<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
        typename gpu<v2048<uint64_t>>::vector_t
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec1,
            int const & p_distance
        ){
            shift_left_elements<<<1,32>>>(p_vec1, p_distance);
            return p_vec1;
        }
    };

    __forceinline__ __device__ uint64_t shift_left_individual_on_device(uint64_t p_vec1_value, uint64_t p_distance) {
        return p_vec1_value << p_distance;
    }

    __global__
    void shift_left_individual_elements(uint64_t* p_vec1, uint64_t* p_distance) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec1[index] = shift_left_individual_on_device(p_vec1[index], p_distance[index]);
    }

    template<>
    struct shift_left_individual<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
        typename gpu<v2048<uint64_t>>::vector_t
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const & p_data,
            typename gpu<v2048<uint64_t>>::vector_t const & p_distance
        ){
            shift_left_individual_elements<<<1,32>>>(p_data, p_distance);
            return p_data;
        }
    };

    __forceinline__ __device__ uint64_t shift_right_on_device(uint64_t p_vec1_value, int p_distance) {
        return p_vec1_value >> p_distance;
    }

    __global__
    void shift_right_elements(uint64_t* p_vec1, int p_distance) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec1[index] = shift_right_on_device(p_vec1[index], p_distance);
    }

    template<>
    struct shift_right<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
        typename gpu<v2048<uint64_t>>::vector_t
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const & p_vec1,
            int const & p_distance
        ){
            shift_right_elements<<<1,32>>>(p_vec1, p_distance);
            return p_vec1;
        }
    };

    __forceinline__ __device__ uint64_t shift_right_individual_on_device(uint64_t p_vec1_value, uint64_t p_distance) {
        return p_vec1_value >> p_distance;
    }

    __global__
    void shift_right_individual_elements(uint64_t* p_vec1, uint64_t* p_distance) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec1[index] = shift_right_individual_on_device(p_vec1[index], p_distance[index]);
    }

    template<>
    struct shift_right_individual<gpu<v2048<uint64_t>>/*, 64*/> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static
        typename gpu<v2048<uint64_t>>::vector_t
        apply(
            typename gpu<v2048<uint64_t>>::vector_t const & p_data,
            typename gpu<v2048<uint64_t>>::vector_t const & p_distance
        ){
            shift_right_individual_elements<<<1,32>>>(p_data, p_distance);
            return p_data;
        }
    };

    __global__
    void count_leadind_zero_device(gpu<v2048<uint64_t>>::mask_t* p_mask, uint8_t* result) {
        *result = __clz(*p_mask);
    }

    template<typename T>
    struct count_leading_zero<gpu<v2048<T>>> {
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static uint8_t*
        apply(
        typename gpu<v2048<U>>::mask_t* const p_mask
        ) {
            uint8_t* result;
            cudaMalloc((void**)&result, sizeof(uint8_t));
            count_leadind_zero_device<<<1,1>>>(p_mask, result);
            return result;

        }
    };

}

#endif /* MORPHSTORE_VECTOR_GPU_PRIMITIVES_CALC_GPU_H */
