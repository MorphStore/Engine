//
// Created by jpietrzyk on 16.05.19.
//

#ifndef MORPHSTORE_LOGIC_H
#define MORPHSTORE_LOGIC_H

#include <vector/general_vector.h>
#include <core/utils/preprocessor.h>

namespace vector {

   template<class VectorExtension, int Granularity>
   struct logic;


   template<class VectorExtension,  int Granularity = VectorExtension::vector_helper_t::size_bit::value>
   MSV_CXX_ATTRIBUTE_FORCE_INLINE
   typename VectorExtension::vector_t
   logical_and(typename VectorExtension::vector_t const & p_In1, typename VectorExtension::vector_t const & p_In2){
      return logic<VectorExtension, Granularity>::logical_and( p_In1, p_In2 );
   }

}
#endif //MORPHSTORE_LOGIC_H
