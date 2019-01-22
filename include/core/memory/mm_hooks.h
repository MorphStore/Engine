/**
 * @file mm_hooks.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MM_HOOKS_H
#define MORPHSTORE_CORE_MEMORY_MM_HOOKS_H

#include "../utils/types.h"

#include <dlfcn.h>
#include <cstdio>

namespace morphstore { namespace memory {
   static void *(*stdlib_malloc_ptr)( size_t ) = nullptr;
   static void *(*stdlib_realloc_ptr)( void *, size_t ) = nullptr;
   static void (*stdlib_free_ptr)( void * ) = nullptr;
} }

static bool init_mem_hooks( void ) {
   morphstore::memory::stdlib_malloc_ptr = ( void *( * )( size_t )) dlsym(RTLD_NEXT, "malloc" );
   morphstore::memory::stdlib_realloc_ptr = ( void *( * )( void *, size_t )) dlsym(RTLD_NEXT, "realloc" );
   morphstore::memory::stdlib_free_ptr = ( void ( * )( void* )) dlsym(RTLD_NEXT, "free" );
   if ( !morphstore::memory::stdlib_malloc_ptr || !morphstore::memory::stdlib_free_ptr || !morphstore::memory::stdlib_realloc_ptr )
      return false;
   return true;
}

#ifdef DEBUG_MALLOC
void * debug_stdlib_malloc( size_t p_AllocSize, const char *file, int line, const char *func ) __THROW {
   fprintf( stderr, "%s - Line %d ( %s ): Kernel Malloc( %zu Byte )\n", file, line, func, p_AllocSize );
   return morphstore::memory::stdlib_malloc_ptr( p_AllocSize );
}
void debug_stdlib_free( void * p_Ptr, const char *file, int line, const char *func ) __THROW {
   fprintf( stderr, "%s - Line %d ( %s ): Kernel Free( %p )\n", file, line, func, p_Ptr );
   morphstore::memory::stdlib_free_ptr( p_Ptr );
}
#  define stdlib_malloc( X ) debug_stdlib_malloc( X, __FILE__, __LINE__, __FUNCTION__ )
#  define stdlib_free( X ) debug_stdlib_free( X, __FILE__, __LINE__, __FUNCTION__ )
#else
#  define stdlib_malloc( X ) stdlib_malloc_ptr( X )
#  define stdlib_free( X ) stdlib_free_ptr( X )
#endif

#endif //MORPHSTORE_CORE_MEMORY_MM_HOOKS_H
