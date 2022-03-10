#ifndef MORPHSTORE_VECTOR_GPU_PRIMITIVES_MANIPULATE_GPU_H
#define MORPHSTORE_VECTOR_GPU_PRIMITIVES_MANIPULATE_GPU_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include "../extension_gpu.h"
#include "../../primitives/manipulate.h"
namespace vectorlib{

    __forceinline__ __device__ uint64_t rotate_on_device(uint64_t p_vec_value, uint64_t index) {
        return __shfl_sync(0xffffffff, p_vec_value, index-1);
    }

    __global__
    void rotate_elements(uint64_t* p_vec) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_vec[index] = rotate_on_device(p_vec[index], index);
    }

    template<typename T>
    struct manipulate<gpu<v2048<T>>, 64> {
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename gpu< v2048< U > >::vector_t
        rotate( gpu< v2048< uint64_t > >::vector_t p_vec ) {
            rotate_elements<<<1,32>>>(p_vec);
        	return p_vec;
        }
    };

}

#endif
