/**
 * @file mm_glob.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MM_GLOB_H
#define MORPHSTORE_CORE_MEMORY_MM_GLOB_H

#ifndef __THROW
#  define __THROW
#endif

#include "mm.h"
#include "mm_hooks.h"
#include "../utils/types.h"
#include "mm_impl.h"


extern "C" {
   void * malloc( size_t p_AllocSize ) __THROW {
      if ( morphstore::memory::stdlib_malloc == nullptr ) {
         init_mem_hooks( );
      }
      return morphstore::memory::query_memory_manager::get_instance( ).allocate( p_AllocSize );
   }

   void free( void *pFreePtr ) __THROW {
      if ( morphstore::memory::stdlib_free == nullptr ) {
         init_mem_hooks( );
      }
      morphstore::memory::query_memory_manager::get_instance( ).deallocate( pFreePtr );
   }
}

#endif //MORPHSTORE_CORE_MEMORY_MM_GLOB_H
