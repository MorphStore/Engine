//
// Created by jpietrzyk on 30.05.19.
//

#ifndef MORPHSTORE_LOGIC_SCALAR_H
#define MORPHSTORE_LOGIC_SCALAR_H
#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/primitives/logic.h>


namespace vectorlib {


   //@todo: The commented part does not compile using clang++
   //type of specialized non-type template argument depends on a template parameter of the partial specialization
   template<typename T>
   struct logic<scalar<v64<T>>, 64/*scalar<v64<T>>::vector_helper_t::size_bit::value*/ > {
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      bitwise_and( typename scalar<v64<T>>::vector_t const & p_In1, typename scalar<v64<T>>::vector_t const & p_In2) {
         return (p_In1 & p_In2 );
      }
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::mask_t
      bitwise_and( typename scalar<v64<T>>::mask_t const & p_In1, typename scalar<v64<T>>::mask_t const & p_In2) {
         return (p_In1 & p_In2 );
      }

      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      typename scalar<v64<T>>::vector_t
      bitwise_or( typename scalar<v64<T>>::vector_t const & p_In1, typename scalar<v64<T>>::vector_t const & p_In2) {
         return (p_In1 | p_In2 );
      }
   };


}
#endif //MORPHSTORE_LOGIC_SCALAR_H
