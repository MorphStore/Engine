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
   void free( void *p_FreePtr ) __THROW {
      if ( morphstore::memory::stdlib_free_ptr == nullptr ) {
         init_mem_hooks( );
      }
      morphstore::memory::query_memory_manager::get_instance( ).deallocate( p_FreePtr );
   }
#else
   void * debug_malloc( size_t p_AllocSize, const char *file, int line, const char *func ) __THROW {
      fprintf( stderr, "[DEBUG]: %s - Line %d ( %s ): MM Malloc( %zu Byte )\n", file, line, func, p_AllocSize );
      if ( morphstore::memory::stdlib_malloc_ptr == nullptr ) {
         init_mem_hooks( );
      }
      return morphstore::memory::query_memory_manager::get_instance( ).allocate( p_AllocSize );
   }
   void debug_free( void *p_FreePtr, const char *file, int line, const char *func ) __THROW {
      fprintf( stderr, "[DEBUG]: %s - Line %d ( %s ): MM Free( %p )\n", file, line, func, p_FreePtr );
      if ( morphstore::memory::stdlib_free_ptr == nullptr ) {
         init_mem_hooks( );
      }
      morphstore::memory::query_memory_manager::get_instance( ).deallocate( p_FreePtr );
   }
#  define malloc( X ) debug_malloc( X, __FILE__, __LINE__, __FUNCTION__ )
#  define free( X ) debug_free( X, __FILE__, __LINE__, __FUNCTION__ )
#endif



}


void * operator new( size_t p_AllocSize ) {
   return malloc( p_AllocSize );
}

void* operator new[]( size_t p_AllocSize ) {
   return malloc( p_AllocSize );
}

void operator delete( void * p_FreePtr ) noexcept {
   free( p_FreePtr );
}
void operator delete( void * p_FreePtr, PPUNUSED size_t p_DeallocSize ) noexcept {
   /**
    * The standard library implementations of the size-aware deallocation functions (5-8) directly call the
    * corresponding size-unaware deallocation functions (1-4).
    * https://en.cppreference.com/w/cpp/memory/new/operator_delete#
    */
   free( p_FreePtr );
}

void operator delete[]( void* p_FreePtr ) noexcept {
   free( p_FreePtr );
}

void operator delete[]( void* p_FreePtr, PPUNUSED size_t p_DeallocSize ) noexcept {
   free( p_FreePtr );
}

#endif //MORPHSTORE_CORE_MEMORY_MM_GLOB_H
