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
 * @file mm_stdlib.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_GLOBAL_MM_STDLIB_H
#define MORPHSTORE_CORE_MEMORY_GLOBAL_MM_STDLIB_H

#include <core/utils/preprocessor.h>


extern "C" {
#if !defined( MSV_DEBUG_MALLOC ) || defined( MSV_NO_LOG )
/**
 * @brief Global replacement for malloc from cstdlib.
 *
 * @details If needed, the defined hook ( morphstore::stdlib_malloc_ptr ) is initialized first. The function
 * forwards the given parameter to morphstore::query_memory_manager::allocate. Thus the query_memory_manager
 * is a thread_local ( thus static ) instance, the malloc call is redirected to the thread specific memory manager.
 *
 * @param p_AllocSize Amount of Bytes which should be allocated.
 *
 * @return Pointer to allocated memory.
 */
void *malloc(size_t p_AllocSize) __THROW {
   return morphstore::query_memory_manager::get_instance().allocate(p_AllocSize);
}

/**
 * @brief Global replacement for free from cstdlib.
 *
 * @details If needed, the defined hook ( morphstore::stdlib_free_ptr ) is initialized first. The function
 * forwards the given parameter to morphstore::query_memory_manager::deallocate. Thus the
 * query_memory_manager is a thread_local ( thus static ) instance, the free call is redirected to the thread
 * specific memory manager.
 *
 * @param p_FreePtr Pointer to allocated memory which should be freed.
 */
void free(void *p_FreePtr) __THROW {
   morphstore::query_memory_manager::get_instance().deallocate(p_FreePtr);
}
#elif defined( MSV_DEBUG_MALLOC )
/**
 * @brief Global helper function for debugging calls to malloc.
 *
 * @details The given parameters (file, line, func) are printed to stderr. If needed, the defined hook
 * ( morphstore::stdlib_malloc_ptr ) is initialized. The function forwards the given parameter to
 * morphstore::query_memory_manager::allocate. Thus the query_memory_manager is a thread_local
 * ( thus static ) instance, the malloc call is redirected to the thread specific memory manager. This functions is
 * only available if the preprocessor variable DEBUG_MALLOC is set. The global malloc( ) call is than "replaced"
 * using a preprocessor macro. This also enables valgrind to actually take the replaced
 * morphstore::query_memory_manager::allocate( ).
 *
 * @param p_AllocSize Amount of Bytes which should be allocated.
 *
 * @return Pointer to allocated memory.
 */
void * debug_malloc( size_t p_AllocSize, const char *file, int line, const char *func ) __THROW {
   void * result = morphstore::query_memory_manager::get_instance( ).allocate( p_AllocSize );
   fprintf( stdout, "[MEM  ]: %s [Managed Malloc] - %p ( %zu Bytes ) [ %s - Line %d ].\n", func, result, p_AllocSize, file, line );
   return result;
}
/**
 * @brief Global helper function for debugging calls to free.
 *
 * @details The given parameters (file, line, func) are printed to stderr. If needed, the defined hook
 * ( morphstore::stdlib_free_ptr ) is initialized. The function forwards the given parameter to
 * morphstore::query_memory_manager::deallocate. Thus the query_memory_manager is a thread_local
 * ( thus static ) instance, the malloc call is redirected to the thread specific memory manager. This functions is
 * only available if the preprocessor variable DEBUG_MALLOC is set. The global free( ) call is than "replaced"
 * using a preprocessor macro. This also enables valgrind to actually take the replaced
 * morphstore::query_memory_manager::deallocate( ).
 *
 * @param p_FreePtr Pointer to allocated memory which should be freed.
 */
void debug_free( void *p_FreePtr, const char *file, int line, const char *func ) __THROW {
   fprintf( stdout, "[MEM  ]: %s [Managed Free] - %p [ %s - Line %d ].\n", func, p_FreePtr, file, line );
   morphstore::query_memory_manager::get_instance( ).deallocate( p_FreePtr );
}
#  define malloc( X ) debug_malloc( X, __FILE__, __LINE__, __FUNCTION__ )
#  define free( X ) debug_free( X, __FILE__, __LINE__, __FUNCTION__ )
#endif
}




#endif //MORPHSTORE_CORE_MEMORY_GLOBAL_MM_STDLIB_H
