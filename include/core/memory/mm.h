/**
 * @file memory_manager.h
 * @brief Defines the memory managment interface ( abstract_memory_manager )
 *
 * @author Johannes Pietrzyk
 * @todo Overthink handle_error( )
 */

#ifndef MORPHSTORE_CORE_MEMORY_MM_H
#define MORPHSTORE_CORE_MEMORY_MM_H


#include "../utils/types.h"


#include <cstdlib>


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
