#ifndef MORPHSTORE_VECTOR_GPU_PRIMITIVES_EXTRACT_GPU_H
#define MORPHSTORE_VECTOR_GPU_PRIMITIVES_EXTRACT_GPU_H

#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include "../extension_gpu.h"
#include "../../primitives/extract.h"
namespace vectorlib{
    __global__
    void extract_elements(uint64_t* p_vec, int index, uint64_t* extracted_value) {
        *extracted_value = p_vec[index];
    }

    template<typename T>
    struct extract<gpu<v2048<T>>,64> {

        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename gpu< v2048< U > >::base_t*
        extract_value( gpu< v2048< uint64_t > >::vector_t p_vec, int idx) {
            uint64_t* extracted_value;
            cudaMalloc((void**)&extracted_value, sizeof(uint64_t));
            extract_elements<<<1,1>>>(p_vec, idx, extracted_value);
            return extracted_value;
        }
    };

}

#endif
