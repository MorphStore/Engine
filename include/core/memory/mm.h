/**
 * @file memory_manager.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MM_H
#define MORPHSTORE_CORE_MEMORY_MM_H


#include "../utils/types.h"


#include <cstdlib>

#ifndef QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE
#define QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE 1_GB
#endif


namespace morphstore { namespace memory {

class abstract_memory_manager {
   public:
      virtual void * allocate( size_t ) = 0;
      virtual void * allocate( abstract_memory_manager *, size_t ) = 0;
      virtual void deallocate( abstract_memory_manager *, void * ) = 0;
      virtual void deallocate( void * ) = 0;
      virtual void handle_error( ) = 0;
};


} }

#endif //MORPHSTORE_CORE_MEMORY_MM_H
