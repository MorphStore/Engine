//
// Created by jpietrzyk on 16.05.19.
//

#ifndef MORPHSTORE_LOGIC_H
#define MORPHSTORE_LOGIC_H

#include <vector/general_vector.h>

namespace vector {

   template<class VectorExtension, int Granularity = VectorExtension::vector_helper_t::size_bit::value>
   struct logic;

}
#endif //MORPHSTORE_LOGIC_H
