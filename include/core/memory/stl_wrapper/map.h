//
// Created by jpietrzyk on 29.06.19.
//

#ifndef MORPHSTORE_CORE_MEMORY_STL_WRAPPTER_MAP_H
#define MORPHSTORE_CORE_MEMORY_STL_WRAPPTER_MAP_H

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_ALLOCATORS_GLOBAL_SCOPE_ALLOCATOR_H
#  error "Perpetual (global scoped) allocator ( allocators/global_scope_allocator.h ) has to be included before all stl_wrapper."
#endif

#include <functional>
#include <map>

namespace morphstore {

   template< typename Key, typename Value >
   typedef std::map<
      Key,
      Value,
      std::less< Key >,
      global_scope_stdlib_allocator< std::pair< const Key, Value > > > map;

}
#endif //MORPHSTORE_CORE_MEMORY_STL_WRAPPTER_MAP_H
