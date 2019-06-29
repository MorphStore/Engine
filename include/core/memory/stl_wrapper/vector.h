//
// Created by jpietrzyk on 29.06.19.
//

#ifndef MORPHSTORE_CORE_MEMORY_STL_WRAPPTER_VECTOR_H
#define MORPHSTORE_CORE_MEMORY_STL_WRAPPTER_VECTOR_H

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_ALLOCATORS_GLOBAL_SCOPE_ALLOCATOR_H
#  error "Perpetual (global scoped) allocator ( allocators/global_scope_allocator.h ) has to be included before all stl_wrapper."
#endif

#include <functional>
#include <vector>

namespace morphstore {

   template< typename Value >
   using vector = std::vector<
      Value,
      global_scope_stdlib_allocator< Value >;

}
#endif //MORPHSTORE_CORE_MEMORY_STL_WRAPPTER_VECTOR_H
