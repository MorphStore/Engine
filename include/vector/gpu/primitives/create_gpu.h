#ifndef MORPHSTORE_VECTOR_GPU_PRIMITIVES_CREATE_GPU_H
#define MORPHSTORE_VECTOR_GPU_PRIMITIVES_CREATE_GPU_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/gpu/extension_gpu.h>
#include <vector/primitives/create.h>

namespace vectorlib{

    __global__
    void set_sequence_elements(uint64_t* d_arr, int a, int b) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        d_arr[index] = a + b * index;
    }

    __global__
    void set1_elements(uint64_t* d_arr, uint64_t a0) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        d_arr[index] = a0;
    }

    __global__
    void set_elements(uint64_t* d_arr, int a0, int a1, int a2, int a3, int a4,
        int a5, int a6, int a7, int a8, int a9, int a10, int a11, int a12,
        int a13, int a14, int a15, int a16, int a17, int a18, int a19, int a20,
        int a21, int a22, int a23, int a24, int a25, int a26, int a27, int a28,
        int a29, int a30, int a31) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        switch (index){
            case 0:  d_arr[index] =  a0; break;
            case 1:  d_arr[index] =  a1; break;
            case 2:  d_arr[index] =  a2; break;
            case 3:  d_arr[index] =  a3; break;
            case 4:  d_arr[index] =  a4; break;
            case 5:  d_arr[index] =  a5; break;
            case 6:  d_arr[index] =  a6; break;
            case 7:  d_arr[index] =  a7; break;
            case 8:  d_arr[index] =  a8; break;
            case 9:  d_arr[index] =  a9; break;
            case 10: d_arr[index] = a10; break;
            case 11: d_arr[index] = a11; break;
            case 12: d_arr[index] = a12; break;
            case 13: d_arr[index] = a13; break;
            case 14: d_arr[index] = a14; break;
            case 15: d_arr[index] = a15; break;
            case 16: d_arr[index] = a16; break;
            case 17: d_arr[index] = a17; break;
            case 18: d_arr[index] = a18; break;
            case 19: d_arr[index] = a19; break;
            case 20: d_arr[index] = a20; break;
            case 21: d_arr[index] = a21; break;
            case 22: d_arr[index] = a22; break;
            case 23: d_arr[index] = a23; break;
            case 24: d_arr[index] = a24; break;
            case 25: d_arr[index] = a25; break;
            case 26: d_arr[index] = a26; break;
            case 27: d_arr[index] = a27; break;
            case 28: d_arr[index] = a28; break;
            case 29: d_arr[index] = a29; break;
            case 30: d_arr[index] = a30; break;
            case 31: d_arr[index] = a31; break;
        }
    }

    template<typename T>
    struct create<gpu<v2048<T>>,64> {

    template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    static typename gpu< v2048< U > >::vector_t
    set(int a0, int a1, int a2, int a3, int a4,
        int a5, int a6, int a7, int a8, int a9, int a10, int a11, int a12,
        int a13, int a14, int a15, int a16, int a17, int a18, int a19, int a20,
        int a21, int a22, int a23, int a24, int a25, int a26, int a27, int a28,
        int a29, int a30, int a31) {
        uint64_t *d_arr;
        cudaMalloc((void**)&d_arr, 32 * sizeof(uint64_t));
        set_elements<<<1,32>>>(d_arr, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31);
        return d_arr;
    }

    template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    static typename gpu< v2048< U > >::vector_t
    set1( uint64_t a0) {
        uint64_t *d_arr;
        cudaMalloc((void**)&d_arr, 32 * sizeof(uint64_t));
        set1_elements<<<1,32>>>(d_arr, a0);
        return d_arr;
    }

    template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    static typename gpu< v2048< U > >::vector_t
        set_sequence( int a, int b) {
        uint64_t *d_arr;
        cudaMalloc((void**)&d_arr, 32 * sizeof(uint64_t));
        set_sequence_elements<<<1,32>>>(d_arr, a, b);
        return d_arr;
      }
   };

}

#endif