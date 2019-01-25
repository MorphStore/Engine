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
   morphstore::memory::stdlib_malloc_ptr = ( void *( * )( size_t )) dlsym(RTLD_NEXT, "malloc" );
   morphstore::memory::stdlib_realloc_ptr = ( void *( * )( void *, size_t )) dlsym(RTLD_NEXT, "realloc" );
   morphstore::memory::stdlib_free_ptr = ( void ( * )( void* )) dlsym(RTLD_NEXT, "free" );
   if ( !morphstore::memory::stdlib_malloc_ptr || !morphstore::memory::stdlib_free_ptr || !morphstore::memory::stdlib_realloc_ptr )
      return false;
   return true;
}

#ifdef MSV_DEBUG_MALLOC
void * debug_stdlib_malloc( size_t p_AllocSize, const char *file, int line, const char *func ) __THROW {
   void * result = morphstore::memory::stdlib_malloc_ptr( p_AllocSize );
   debug( file, " - Line ", line, " ( ", func, " ): Kernel Malloc( ", p_AllocSize, " Bytes ) ");
   return result;
}
void debug_stdlib_free( void * p_Ptr, const char *file, int line, const char *func ) __THROW {
   debug( file, " - Line ", line, " ( ", func, " ): Kernel Free( ", p_Ptr, " ) ");
   morphstore::memory::stdlib_free_ptr( p_Ptr );
}
#  define stdlib_malloc( X ) debug_stdlib_malloc( X, __FILE__, __LINE__, __FUNCTION__ )
#  define stdlib_free( X ) debug_stdlib_free( X, __FILE__, __LINE__, __FUNCTION__ )
#else
#  define stdlib_malloc( X ) stdlib_malloc_ptr( X )
#  define stdlib_free( X ) stdlib_free_ptr( X )
#endif

#endif //MORPHSTORE_CORE_MEMORY_MM_HOOKS_H
