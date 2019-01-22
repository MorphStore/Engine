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
#include <cstdio>

extern "C" {
#ifndef DEBUG_MALLOC
   void * malloc( size_t p_AllocSize ) __THROW {
      if ( morphstore::memory::stdlib_malloc_ptr == nullptr ) {
         init_mem_hooks( );
      }
      return morphstore::memory::query_memory_manager::get_instance( ).allocate( p_AllocSize );
   }
#else
   void * debug_malloc( size_t p_AllocSize, const char *file, int line, const char *func ) __THROW {
      fprintf( stderr, "%s - Line %d ( %s ): MM Malloc( %zu Byte )\n", file, line, func, p_AllocSize );
      if ( morphstore::memory::stdlib_malloc_ptr == nullptr ) {
         init_mem_hooks( );
      }
      return morphstore::memory::query_memory_manager::get_instance( ).allocate( p_AllocSize );
   }
#  define malloc( X ) debug_malloc( X, __FILE__, __LINE__, __FUNCTION__ )
#endif

   void free( void *pFreePtr ) __THROW {
      if ( morphstore::memory::stdlib_free_ptr == nullptr ) {
         init_mem_hooks( );
      }
      morphstore::memory::query_memory_manager::get_instance( ).deallocate( pFreePtr );
   }
}



#endif //MORPHSTORE_CORE_MEMORY_MM_GLOB_H
