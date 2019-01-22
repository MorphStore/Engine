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

namespace morphstore { namespace memory {
static void *(*stdlib_malloc)( size_t ) = nullptr;
static void *(*stdlib_realloc)( void *, size_t ) = nullptr;
static void (*stdlib_free)( void * ) = nullptr;
} }

static bool init_mem_hooks( void ) {
   morphstore::memory::stdlib_malloc = ( void *( * )( size_t )) dlsym(RTLD_NEXT, "malloc" );
   morphstore::memory::stdlib_realloc = ( void *( * )( void *, size_t )) dlsym(RTLD_NEXT, "realloc" );
   morphstore::memory::stdlib_free = ( void ( * )( void* )) dlsym(RTLD_NEXT, "free" );
   if ( !morphstore::memory::stdlib_malloc || !morphstore::memory::stdlib_free || !morphstore::memory::stdlib_realloc )
      return false;
   return true;
}

#endif //MORPHSTORE_CORE_MEMORY_MM_HOOKS_H
