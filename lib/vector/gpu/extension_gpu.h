#ifndef MORPHSTORE_VECTOR_GPU_EXTENSION_GPU_H
#define MORPHSTORE_VECTOR_GPU_EXTENSION_GPU_H

#include <cstdint>
#include <type_traits>
#include "../vector_extension_structs.h"

namespace vectorlib{
   template<class VectorReg>
   struct gpu;

   template<typename T>
   struct gpu< v2048< T > > {
		static_assert(std::is_arithmetic<T>::value, "Base type of vector register has to be arithmetic.");
    	using vector_helper_t = v2048<T>;
    	using base_t = typename vector_helper_t::base_t;
      	using vector_t =
        	typename std::conditional<
            (1==1) == std::is_integral<T>::value,    // if T is integer
            uint64_t*,
            typename std::conditional<
               (1==1) == std::is_same<float, T>::value, // else if T is float
               float*,
               double*                       // else [T == double]
            >::type
        >::type;
		using size = std::integral_constant<size_t, sizeof(vector_t)>;
		using mask_t = uint32_t;
   };
}

#endif //MORPHSTORE_VECTOR_GPU_EXTENSION_GPU_H
