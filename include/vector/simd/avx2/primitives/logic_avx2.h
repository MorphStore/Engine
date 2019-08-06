//
// Created by jpietrzyk on 20.05.19.
//

#ifndef MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_LOGIC_AVX2_H
#define MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_LOGIC_AVX2_H
#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/primitives/logic.h>


namespace vectorlib {


   template<typename T>
   struct logic<avx2<v256<T>>, avx2<v256<T>>::vector_helper_t::size_bit::value > {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<T>>::vector_t
      bitwise_and( typename avx2<v256<T>>::vector_t const & p_In1, typename avx2<v256<T>>::vector_t const & p_In2) {
         return _mm256_and_si256( p_In1, p_In2 );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx2<v256<T>>::vector_t
      bitwise_or( typename avx2<v256<T>>::vector_t const & p_In1, typename avx2<v256<T>>::vector_t const & p_In2) {
         return _mm256_or_si256( p_In1, p_In2 );
      }
   };


}
#endif //MORPHSTORE_VECTOR_SIMD_AVX2_PRIMITIVES_LOGIC_AVX2_H
