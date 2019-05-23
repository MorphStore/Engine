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
#include <core/memory/management/abstract_mm.h>

#include <core/memory/management/mmap_mm.h>
#include <core/memory/management/paged_mm.h>

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
   //return morphstore::query_memory_manager::get_instance().allocate(p_AllocSize);
    //const size_t CACHE_LINE_SIZE = 64;
    size_t abs_needed_size = p_AllocSize + sizeof(morphstore::ObjectInfo);

    if (abs_needed_size > morphstore::ALLOCATION_SIZE) { /* already allocates object info for convenience, aligned to chunksize */
        return morphstore::mmap_memory_manager::getInstance().allocateLarge(p_AllocSize);
    }
    else if (abs_needed_size > (( 1l<<14 ) - sizeof(morphstore::PageHeader) )) {
        return morphstore::mmap_memory_manager::getInstance().allocate(abs_needed_size);
    }
    else {
        return morphstore::paged_memory_manager::getGlobalInstance().allocate(abs_needed_size);
    }
}

void *realloc( void * p_Ptr, size_t p_AllocSize ) __THROW {
   //return morphstore::query_memory_manager::get_instance().reallocate(p_Ptr, p_AllocSize);
    morphstore::ObjectInfo* info = reinterpret_cast<morphstore::ObjectInfo*>( reinterpret_cast<uint64_t>(p_Ptr) - sizeof(morphstore::ObjectInfo));

    if (info->size > morphstore::ALLOCATION_SIZE) {
        return morphstore::mmap_memory_manager::getInstance().reallocate(p_Ptr, p_AllocSize);
    }
    else if (info->size > (( morphstore::DB_PAGE_SIZE - sizeof(morphstore::PageHeader) ))) {
        return morphstore::mmap_memory_manager::getInstance().reallocate(info, p_AllocSize + sizeof(morphstore::ObjectInfo));
    }
    else {
        return morphstore::paged_memory_manager::getGlobalInstance().reallocate(info, p_AllocSize + sizeof(morphstore::ObjectInfo));
    }
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
    morphstore::ObjectInfo* info = reinterpret_cast<morphstore::ObjectInfo*>( reinterpret_cast<uint64_t>(p_FreePtr) - sizeof(morphstore::ObjectInfo));

    if (info->size > morphstore::ALLOCATION_SIZE) {
        morphstore::mmap_memory_manager::getInstance().deallocate(p_FreePtr);
    }
    else if (info->size > (( morphstore::DB_PAGE_SIZE - sizeof(morphstore::PageHeader) ))) {
        morphstore::mmap_memory_manager::getInstance().deallocate(info);
    }
    else {
        morphstore::paged_memory_manager::getGlobalInstance().deallocate(info);
    }
   /*if( MSV_CXX_ATTRIBUTE_LIKELY( morphstore::query_memory_manager_state_helper::get_instance( ).is_alive( ) ) )
      morphstore::query_memory_manager::get_instance().deallocate(p_FreePtr);
   else
      morphstore::stdlib_free_ptr( p_FreePtr );*/
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
   void * result = malloc(p_AllocSize);
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
   /*if( MSV_CXX_ATTRIBUTE_LIKELY( morphstore::query_memory_manager_state_helper::get_instance( ).is_alive( ) ) )
      morphstore::query_memory_manager::get_instance().deallocate(p_FreePtr);
   else
      morphstore::stdlib_free_ptr( p_FreePtr );*/
   free(p_FreePtr);
}
#  define malloc( X ) debug_malloc( X, __FILE__, __LINE__, __FUNCTION__ )
#  define free( X ) debug_free( X, __FILE__, __LINE__, __FUNCTION__ )
#endif
}




#endif //MORPHSTORE_CORE_MEMORY_GLOBAL_MM_STDLIB_H
