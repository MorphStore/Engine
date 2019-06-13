//
// Created by jpietrzyk on 20.05.19.
//

#ifndef MORPHSTORE_LOGIC_AVX512_H
#define MORPHSTORE_LOGIC_AVX512_H
#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx512/extension_avx512.h>
#include <vector/primitives/logic.h>


namespace vector {


   template<typename T>
   struct logic<avx512<v512<T>>, avx512<v512<T>>::vector_helper_t::size_bit::value > {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<T>>::vector_t
      bitwise_and( typename avx512<v512<T>>::vector_t const & p_In1, typename avx512<v512<T>>::vector_t const & p_In2) {
         return _mm512_and_si512( p_In1, p_In2 );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<T>>::vector_t
      bitwise_or( typename avx512<v512<T>>::vector_t const & p_In1, typename avx512<v512<T>>::vector_t const & p_In2) {
         return _mm512_or_si512( p_In1, p_In2 );
      }
   };


}
#endif //MORPHSTORE_LOGIC_AVX512_H
