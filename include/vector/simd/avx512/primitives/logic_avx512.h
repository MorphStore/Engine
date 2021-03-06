//
// Created by jpietrzyk on 20.05.19.
//

#ifndef MORPHSTORE_VECTOR_SIMD_AVX512_PRIMITIVES_LOGIC_AVX512_H
#define MORPHSTORE_VECTOR_SIMD_AVX512_PRIMITIVES_LOGIC_AVX512_H
#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/simd/avx512/extension_avx512.h>
#include <vector/primitives/logic.h>


namespace vectorlib {


   template<typename T>
   struct logic<avx512<v512<T>>, avx512<v512<T>>::vector_helper_t::size_bit::value > {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::vector_t
      bitwise_and( typename avx512<v512<T>>::vector_t const & p_In1, typename avx512<v512<T>>::vector_t const & p_In2) {
         return _mm512_and_si512( p_In1, p_In2 );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<uint64_t>>::mask_t
      bitwise_and( typename avx512<v512<uint64_t>>::mask_t const & p_In1, typename avx512<v512<T>>::mask_t const & p_In2) {
         return _kand_mask8( p_In1, p_In2 );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v512<T>>::vector_t
      bitwise_or( typename avx512<v512<T>>::vector_t const & p_In1, typename avx512<v512<T>>::vector_t const & p_In2) {
         return _mm512_or_si512( p_In1, p_In2 );
      }
   };

   template<typename T>
   struct logic<avx512<v256<T>>, avx512<v256<T>>::vector_helper_t::size_bit::value > {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<uint64_t>>::vector_t
      bitwise_and( typename avx512<v256<T>>::vector_t const & p_In1, typename avx512<v256<T>>::vector_t const & p_In2) {
         return _mm256_and_si256( p_In1, p_In2 );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v256<T>>::vector_t
      bitwise_or( typename avx512<v256<T>>::vector_t const & p_In1, typename avx512<v256<T>>::vector_t const & p_In2) {
         return _mm256_or_si256( p_In1, p_In2 );
      }
   };

   
   template<typename T>
   struct logic<avx512<v128<T>>, avx512<v128<T>>::vector_helper_t::size_bit::value > {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<uint64_t>>::vector_t
      bitwise_and( typename avx512<v128<T>>::vector_t const & p_In1, typename avx512<v128<T>>::vector_t const & p_In2) {
         return _mm_and_si128( p_In1, p_In2 );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename avx512<v128<T>>::vector_t
      bitwise_or( typename avx512<v128<T>>::vector_t const & p_In1, typename avx512<v128<T>>::vector_t const & p_In2) {
         return _mm_or_si128( p_In1, p_In2 );
      }
   };


}
#endif //MORPHSTORE_VECTOR_SIMD_AVX512_PRIMITIVES_LOGIC_AVX512_H
