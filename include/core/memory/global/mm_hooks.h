/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
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
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_GLOBAL_MM_HOOKS_H
#define MORPHSTORE_CORE_MEMORY_GLOBAL_MM_HOOKS_H

#include "../../utils/preprocessor.h"

//#define MSV_DEBUG_MALLOC

#include <dlfcn.h>
#include <cstdio>

namespace morphstore {
   static void *(*stdlib_malloc_ptr)( size_t ) = nullptr;
   static void *(*stdlib_realloc_ptr)( void *, size_t ) = nullptr;
   static void (*stdlib_free_ptr)( void * ) = nullptr;
#ifdef MSV_MEMORY_LEAK_CHECK
   void leak_detector_malloc_called( void * const, size_t const );
   void leak_detector_free_called( void const * const );
#endif
}

static bool init_mem_hooks( void ) {
   morphstore::stdlib_malloc_ptr = ( void *( * )( size_t )) dlsym(RTLD_NEXT, "malloc" );
   morphstore::stdlib_realloc_ptr = ( void *( * )( void *, size_t )) dlsym(RTLD_NEXT, "realloc" );
   morphstore::stdlib_free_ptr = ( void ( * )( void* )) dlsym(RTLD_NEXT, "free" );
   if ( !morphstore::stdlib_malloc_ptr || !morphstore::stdlib_free_ptr || !morphstore::stdlib_realloc_ptr ) {
      return false;
   }
   return true;
}


#if defined( MSV_DEBUG_MALLOC ) && !defined( MSV_NO_LOG )
void * debug_stdlib_malloc( size_t p_AllocSize, const char *file, int line, const char *func ) __THROW {
   void * result = morphstore::stdlib_malloc_ptr( p_AllocSize );
#ifdef MSV_MEMORY_LEAK_CHECK
   morphstore::leak_detector_malloc_called( result, p_AllocSize );
#endif
   fprintf( stdout, "[MEM  ]: %s [Kernel Malloc] - %p ( %zu Bytes ) [ %s - Line %d ].\n", func, result, p_AllocSize, file, line );
   return result;
}
void * debug_stdlib_realloc( void * p_Ptr, size_t p_AllocSize, const char *file, int line, const char *func ) __THROW {
   void * result = morphstore::stdlib_realloc_ptr( p_Ptr, p_AllocSize );
#ifdef MSV_MEMORY_LEAK_CHECK
   morphstore::leak_detector_malloc_called( result, p_AllocSize );
   morphstore::leak_detector_free_called( p_Ptr );
#endif
   fprintf( stdout, "[MEM  ]: %s [Kernel Realloc] - %p -> %p ( %zu Bytes ) [ %s - Line %d ].\n", func, p_Ptr, result, p_AllocSize, file, line );
   return result;
}
void debug_stdlib_free( void * p_Ptr, const char *file, int line, const char *func ) __THROW {
#ifdef MSV_MEMORY_LEAK_CHECK
   morphstore::leak_detector_free_called( p_Ptr );
#endif
   fprintf( stdout, "[MEM  ]: %s [Kernel Free] - %p [ %s - Line %d ].\n", func, p_Ptr, file, line );
   morphstore::stdlib_free_ptr( p_Ptr );
}
#     define stdlib_malloc( X ) debug_stdlib_malloc( X, __FILE__, __LINE__, __FUNCTION__ )
#     define stdlib_realloc( X ) debug_stdlib_realloc( X, __FILE__, __LINE__, __FUNCTION__ )
#     define stdlib_free( X ) debug_stdlib_free( X, __FILE__, __LINE__, __FUNCTION__ )
#  elif defined( MSV_MEMORY_LEAK_CHECK )
void * mem_leak_stdlib_malloc( size_t p_AllocSize ) __THROW {
   void * result = morphstore::stdlib_malloc_ptr( p_AllocSize );
   morphstore::leak_detector_malloc_called( result, p_AllocSize );
   return result;
}
void * mem_leak_stdlib_realloc( void * p_Ptr, size_t p_AllocSize ) __THROW {
   void * result = morphstore::stdlib_realloc_ptr( p_Ptr, p_AllocSize );
   morphstore::leak_detector_malloc_called( result, p_AllocSize );
   morphstore::leak_detector_free_called( p_Ptr );
   return result;
}
void mem_leak_stdlib_free( void * p_Ptr ) __THROW {
   morphstore::leak_detector_free_called( p_Ptr );
   morphstore::stdlib_free_ptr( p_Ptr );
}
#     define stdlib_malloc( X ) mem_leak_stdlib_malloc( X )
#     define stdlib_realloc( X ) mem_leak_stdlib_realloc( X )
#     define stdlib_free( X ) mem_leak_stdlib_free( X )
#  else
#     define stdlib_malloc( X ) stdlib_malloc_ptr( X )
#     define stdlib_realloc( X ) stdlib_realloc_ptr( X )
#     define stdlib_free( X ) stdlib_free_ptr( X )
#  endif

extern "C" {
   void *malloc(size_t) __THROW;
   void *realloc(void *, size_t) __THROW;
   void free(void *) __THROW;
}
/**
 * @brief Global replacement for operator new.
 *
 * @details This method is implemented for convenience reasons. The parameter is forwarded to malloc.
 *
 * @param p_AllocSize Amount of Bytes which should be allocated.
 *
 * @return Pointer to allocated memory.
 */
void * operator new( size_t p_AllocSize ) {
   return malloc( p_AllocSize );
}
/**
 * @brief Global replacement for operator new[].
 *
 * @details This method is implemented for convenience reasons. The parameter is forwarded to malloc.
 *
 * @param p_AllocSize Amount of Bytes which should be allocated.
 *
 * @return Pointer to allocated memory.
 */
void* operator new[]( size_t p_AllocSize ) {
   return malloc( p_AllocSize );
}

/**
 * @brief Global replacement for operator delete.
 *
 * @details This method is implemented for convenience reasons. The parameter is forwarded to free.
 *
 * @param p_FreePtr Pointer to allocated memory which should be freed.
 */
void operator delete( void * p_FreePtr ) noexcept {
   free( p_FreePtr );
}

/**
 * @brief Global replacement for operator delete.
 *
 * @details This method is implemented for convenience reasons. The parameter is forwarded to free.
 * "The standard library implementations of the size-aware deallocation functions (5-8) directly call the
 * corresponding size-unaware deallocation functions (1-4)."
 * (https://en.cppreference.com/w/cpp/memory/new/operator_delete)
 *
 * @param p_FreePtr Pointer to allocated memory which should be freed.
 * @param p_DeallocSize Unused (see details)
 */
void operator delete( void * p_FreePtr, MSV_CXX_ATTRIBUTE_PPUNUSED size_t p_DeallocSize ) noexcept {
   free( p_FreePtr );
}

/**
 * @brief Global replacement for operator delete[].
 *
 * @details This method is implemented for convenience reasons. The parameter is forwarded to free.
 *
 * @param p_FreePtr Pointer to allocated memory which should be freed.
 */
void operator delete[]( void* p_FreePtr ) noexcept {
   free( p_FreePtr );
}

/**
 * @brief Global replacement for operator delete[].
 *
 * @details This method is implemented for convenience reasons. The parameter is forwarded to free.
 * "The standard library implementations of the size-aware deallocation functions (5-8) directly call the
 * corresponding size-unaware deallocation functions (1-4)."
 * (https://en.cppreference.com/w/cpp/memory/new/operator_delete)
 *
 * @param p_FreePtr Pointer to allocated memory which should be freed.
 * @param p_DeallocSize Unused (see details)
 */
void operator delete[]( void* p_FreePtr, MSV_CXX_ATTRIBUTE_PPUNUSED size_t p_DeallocSize ) noexcept {
   free( p_FreePtr );
}

#endif //MORPHSTORE_CORE_MEMORY_GLOBAL_MM_HOOKS_H
