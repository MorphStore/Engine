/**********************************************************************************************
 * Copyright (C) 2019 by Johannes Pietrzyk                                                    *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/


/**
 * @file mm_hooks.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MM_HOOKS_H
#define MORPHSTORE_CORE_MEMORY_MM_HOOKS_H

#ifndef MORPHSTORE_CORE_MEMORY_MM_GLOB_H
#  error "mm_hooks.h has to be included AFTER mm_glob.h"
#endif

#ifndef MSV_NO_SELFMANAGED_MEMORY

#include "../utils/types.h"
#include "../utils/logger.h"

#include <dlfcn.h>
#include <cstdio>

namespace morphstore { namespace memory {
   static void *(*stdlib_malloc_ptr)( size_t ) = nullptr;
   static void *(*stdlib_realloc_ptr)( void *, size_t ) = nullptr;
   static void (*stdlib_free_ptr)( void * ) = nullptr;
} }

static bool init_mem_hooks( void ) {
   trace( "[] - Initializing hooks." );
   morphstore::memory::stdlib_malloc_ptr = ( void *( * )( size_t )) dlsym(RTLD_NEXT, "malloc" );
   morphstore::memory::stdlib_realloc_ptr = ( void *( * )( void *, size_t )) dlsym(RTLD_NEXT, "realloc" );
   morphstore::memory::stdlib_free_ptr = ( void ( * )( void* )) dlsym(RTLD_NEXT, "free" );
   if ( !morphstore::memory::stdlib_malloc_ptr || !morphstore::memory::stdlib_free_ptr || !morphstore::memory::stdlib_realloc_ptr ) {
      trace( "[] - Could not initialize hooks." );
      return false;
   }
   trace( "[] - Hooks initialized.");
   return true;
}

#  if defined( MSV_DEBUG_MALLOC ) && !defined( MSV_NO_LOG )
void * debug_stdlib_malloc( size_t p_AllocSize, const char *file, int line, const char *func ) __THROW {
   void * result = morphstore::memory::stdlib_malloc_ptr( p_AllocSize );
   info( "Kernel Malloc:", result, "Size =", p_AllocSize, "Bytes [", file, "- Line", line, "(", func, ") ]");
   return result;
}
void debug_stdlib_free( void * p_Ptr, const char *file, int line, const char *func ) __THROW {
   info( "Kernel Free  :", p_Ptr, "[", file, "- Line", line, "(", func, ") ]");
   morphstore::memory::stdlib_free_ptr( p_Ptr );
}
#     define stdlib_malloc( X ) debug_stdlib_malloc( X, __FILE__, __LINE__, __FUNCTION__ )
#     define stdlib_free( X ) debug_stdlib_free( X, __FILE__, __LINE__, __FUNCTION__ )
#  else
#     define stdlib_malloc( X ) stdlib_malloc_ptr( X )
#     define stdlib_free( X ) stdlib_free_ptr( X )
#  endif

#endif // !define(MSV_NO_SELFMANAGED_MEMORY)
#endif //MORPHSTORE_CORE_MEMORY_MM_HOOKS_H
