#ifndef MORPHSTORE_VECTOR_GPU_PRIMITIVES_IO_GPU_H
#define MORPHSTORE_VECTOR_GPU_PRIMITIVES_IO_GPU_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include "../extension_gpu.h"
#include "../../primitives/io.h"
namespace vectorlib{

    __global__
    void load_elements(uint64_t* d_arr, uint64_t* p_DataPtr) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        d_arr[index] = p_DataPtr[index];
    }

    __global__
    void store_elements(uint64_t* p_DataPtr, uint64_t* p_vec) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        p_DataPtr[index] = p_vec[index];
    }

    template<typename T, int IOGranularity>
    struct io<gpu<v2048<T>>,iov::ALIGNED, IOGranularity> { //gives out a vector with 32 elements starting from p_DataPtr
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename gpu< v2048< U > >::vector_t
        load( U * p_DataPtr ) {
            uint64_t *d_arr;
            cudaMalloc((void**)&d_arr, 32 * sizeof(uint64_t));
            load_elements<<<1,32>>>(d_arr, p_DataPtr);
            return d_arr;
        }

        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static void
        store( U * p_DataPtr, gpu< v2048< int > >::vector_t p_vec ) {
            store_elements<<<1,32>>>(p_DataPtr, p_vec);
            return;
        }
    };

    __global__
    void compressstore_elements(uint64_t* p_DataPtr, uint64_t* p_vec, int mask) {
        //uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        int j = 0;
        for(int i=0; i<32; i++){
            if((mask >> i) & 1){
                p_DataPtr[j] = p_vec[i];
                j++;
            }
        }
    }

    template<typename T, int IOGranularity>
    struct io<gpu<v2048<T>>,iov::UNALIGNED, IOGranularity> {
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename gpu< v2048< U > >::vector_t
        load( U * p_DataPtr ) {
            uint64_t *d_arr;
            cudaMalloc((void**)&d_arr, 32 * sizeof(uint64_t));
            load_elements<<<1,32>>>(d_arr, p_DataPtr);
            return d_arr;
        }

        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static void
        store( U * p_DataPtr, gpu< v2048< int > >::vector_t p_vec ) {
            store_elements<<<1,32>>>(p_DataPtr, p_vec);
            return;
        }

        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static void
        compressstore( U * p_DataPtr,  typename gpu< v2048< U > >::vector_t p_vec, int mask ) {
            compressstore_elements<<<1,1>>>(p_DataPtr, p_vec, mask);
            return ;
        }
    };

    __global__
    void gather_elements(uint64_t* d_arr, uint64_t* p_DataPtr, uint64_t* p_vec, int scale) {
        uint64_t index = threadIdx.x + blockIdx.x * blockDim.x;
        d_arr[index] = p_DataPtr[p_vec[index * scale]];
    }

    template<typename T, int IOGranularity, int Scale>
    struct gather_t<gpu<v2048<T>>, IOGranularity, Scale> {
        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename gpu< v2048< U > >::vector_t
        apply( U * p_DataPtr,  gpu< v2048< uint64_t > >::vector_t p_vec ) {
            uint64_t *d_arr;
            cudaMalloc((void**)&d_arr, 32 * sizeof(uint64_t));
            gather_elements<<<1,32>>>(d_arr, p_DataPtr, p_vec, Scale);
            return d_arr;
        }

    };

}

#endif
