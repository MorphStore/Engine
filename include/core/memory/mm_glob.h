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
 * @file mm_glob.h
 * @brief Overrides for global memory access functions.
 *
 * @details Replaces malloc, free, new, new[], delete, delete[]. If the preprocessor variable DEBUG_MALLOC is set,
 * malloc and free will print the caller file, line and function to stderr. Malloc and free inits the hooks defined in
 * mm_hooks.h at first if they are not initialized yet. Afterwards the allocate is called on the thread_local
 * query_memory_manager. Free does the same as malloc but instead calling allocate, deallocate is called.
 *
 * @author Johannes Pietrzyk
 *
 * @todo Get rid of repeatedly checking the hook pointer ( stdlib_malloc_ptr ).
 */

#ifndef MORPHSTORE_CORE_MEMORY_MM_GLOB_H
#  define MORPHSTORE_CORE_MEMORY_MM_GLOB_H

#include "../utils/preprocessor.h"
#include "../utils/types.h"
#include "../utils/logger.h"




#  ifdef MSV_NO_SELFMANAGED_MEMORY
#     ifdef MSV_DEBUG_MALLOC
#        ifndef __THROW
#           define __THROW
#        endif
   void * debug_malloc( size_t p_AllocSize, const char *file, int line, const char *func ) __THROW {
      void * result = malloc( p_AllocSize );
      trace( "Kernel Malloc: ", result, " ( ", p_AllocSize, " Bytes ) [ ", file, " - Line ", line, " ( ", func, " ) ]" );
      return result;
   }
   void debug_free( void *p_FreePtr, const char *file, int line, const char *func ) __THROW {
      trace( "Kernel Free: ", p_FreePtr, " [ ", file, " - Line ", line, " ( ", func, " ) ]");
      free( p_FreePtr );
   }
#        define malloc( X ) debug_malloc( X, __FILE__, __LINE__, __func__ )
#        define free( X ) debug_free( X, __FILE__, __LINE__, __func__ )

   /** replacing the new and delete operator seems to be a bit tricky. All we can do is overloading them.
    * By overloading we can not change the signature of the call to get an appropriate file string and a corresponding
    * line.
    * Nevertheless we are not alone with this goal. Have a look into this blog:
    * https://blogs.msdn.microsoft.com/calvin_hsia/2009/01/19/overload-operator-new-to-detect-memory-leaks/
    * @todo: Implement and test this.
    */
   void * operator new( size_t p_AllocSize ) {
      return malloc( p_AllocSize );
   }
   void* operator new[]( size_t p_AllocSize ) {
      return malloc( p_AllocSize );
   }
   void operator delete( void * p_FreePtr ) noexcept {
      free( p_FreePtr );
   }
   void operator delete( void * p_FreePtr, MSV_PPUNUSED size_t p_DeallocSize ) noexcept {
      free( p_FreePtr );
   }
   void operator delete[]( void* p_FreePtr ) noexcept {
      free( p_FreePtr );
   }
   void operator delete[]( void* p_FreePtr, MSV_PPUNUSED size_t p_DeallocSize ) noexcept {
      free( p_FreePtr );
   }
#     endif
#  else  // ! MSV_NO_SELFMANAGED_MEMORY

#     ifndef __THROW
#        define __THROW
#     endif

#     include "mm.h"
#     include "mm_hooks.h"
#     include "mm_impl.h"
#     include <cstdio>


extern "C" {
#     ifndef MSV_DEBUG_MALLOC
   /**
    * @brief Global replacement for malloc from cstdlib.
    *
    * @details If needed, the defined hook ( morphstore::memory::stdlib_malloc_ptr ) is initialized first. The function
    * forwards the given parameter to morphstore::memory::query_memory_manager::allocate. Thus the query_memory_manager
    * is a thread_local ( thus static ) instance, the malloc call is redirected to the thread specific memory manager.
    *
    * @param p_AllocSize Amount of Bytes which should be allocated.
    *
    * @return Pointer to allocated memory.
    */
   void * malloc( size_t p_AllocSize ) __THROW {
      return morphstore::memory::query_memory_manager::get_instance( ).allocate( p_AllocSize );
   }

   /**
    * @brief Global replacement for free from cstdlib.
    *
    * @details If needed, the defined hook ( morphstore::memory::stdlib_free_ptr ) is initialized first. The function
    * forwards the given parameter to morphstore::memory::query_memory_manager::deallocate. Thus the
    * query_memory_manager is a thread_local ( thus static ) instance, the free call is redirected to the thread
    * specific memory manager.
    *
    * @param p_FreePtr Pointer to allocated memory which should be freed.
    */
   void free( void *p_FreePtr ) __THROW {
      morphstore::memory::query_memory_manager::get_instance( ).deallocate( p_FreePtr );
   }
#     else
   /**
    * @brief Global helper function for debugging calls to malloc.
    *
    * @details The given parameters (file, line, func) are printed to stderr. If needed, the defined hook
    * ( morphstore::memory::stdlib_malloc_ptr ) is initialized. The function forwards the given parameter to
    * morphstore::memory::query_memory_manager::allocate. Thus the query_memory_manager is a thread_local
    * ( thus static ) instance, the malloc call is redirected to the thread specific memory manager. This functions is
    * only available if the preprocessor variable DEBUG_MALLOC is set. The global malloc( ) call is than "replaced"
    * using a preprocessor macro. This also enables valgrind to actually take the replaced
    * morphstore::memory::query_memory_manager::allocate( ).
    *
    * @param p_AllocSize Amount of Bytes which should be allocated.
    *
    * @return Pointer to allocated memory.
    */
   void * debug_malloc( size_t p_AllocSize, const char *file, int line, const char *func ) __THROW {
      debug( "Managed Malloc:", p_AllocSize, "Bytes (", file, " - Line ", line, " ( ", func, " )");
      return morphstore::memory::query_memory_manager::get_instance( ).allocate( p_AllocSize );
   }
   /**
    * @brief Global helper function for debugging calls to free.
    *
    * @details The given parameters (file, line, func) are printed to stderr. If needed, the defined hook
    * ( morphstore::memory::stdlib_free_ptr ) is initialized. The function forwards the given parameter to
    * morphstore::memory::query_memory_manager::deallocate. Thus the query_memory_manager is a thread_local
    * ( thus static ) instance, the malloc call is redirected to the thread specific memory manager. This functions is
    * only available if the preprocessor variable DEBUG_MALLOC is set. The global free( ) call is than "replaced"
    * using a preprocessor macro. This also enables valgrind to actually take the replaced
    * morphstore::memory::query_memory_manager::deallocate( ).
    *
    * @param p_FreePtr Pointer to allocated memory which should be freed.
    */
   void debug_free( void *p_FreePtr, const char *file, int line, const char *func ) __THROW {
      debug( file, " - Line ", line, " ( ", func, " ): MM Free( ", p_FreePtr, " ) ");
      morphstore::memory::query_memory_manager::get_instance( ).deallocate( p_FreePtr );
   }
#        define malloc( X ) debug_malloc( X, __FILE__, __LINE__, __FUNCTION__ )
#        define free( X ) debug_free( X, __FILE__, __LINE__, __FUNCTION__ )
#     endif
/**
 * @brief Wrapper for glibc implementation of thread_atexit
 * @param func
 * @param obj
 * @param dso_symbol
 * @return
 */
//int __cxa_thread_atexit( void ( * func )( ), void * obj, void * dso_symbol ) {
//   int __cxa_thread_atexit_impl( void ( * )( ), void *, void * );
//   return __cxa_thread_atexit_impl( func, obj, dso_symbol );
//}

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
void operator delete( void * p_FreePtr, MSV_PPUNUSED size_t p_DeallocSize ) noexcept {
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
void operator delete[]( void* p_FreePtr, MSV_PPUNUSED size_t p_DeallocSize ) noexcept {
   free( p_FreePtr );
}


#  endif





#endif //MORPHSTORE_CORE_MEMORY_MM_GLOB_H
