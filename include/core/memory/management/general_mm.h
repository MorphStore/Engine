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
 * @file general_mm.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_GENERAL_MM_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_GENERAL_MM_H

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_ABSTRACT_MM_H
#  error "Abstract memory manager ( management/abstract_mm.h ) has to be included before general memory manager."
#endif

#ifndef MORPHSTORE_CORE_MEMORY_GLOBAL_MM_HOOKS_H
#  error "Memory Hooks ( global/mm_hooks.h ) has to be included before general memory manager."
#endif

#include <core/utils/basic_types.h>
#include <core/utils/helper.h>
#include <core/utils/preprocessor.h>
#include <numa.h>

#include <core/memory/management/utils/memory_bin_handler.h>

namespace morphstore {

class general_memory_manager : public abstract_memory_manager {
   public:
      static general_memory_manager &get_instance(void) {
         trace( "[General Memory Manager] - IN.  ( void )." );
         static general_memory_manager instance;
         trace( "[General Memory Manager] - OUT. ( Instance: ", &instance, " )." );
         return instance;
      }

      general_memory_manager(general_memory_manager const &) = delete;

      general_memory_manager &operator=(general_memory_manager const &) = delete;

      virtual ~general_memory_manager(void) {
         trace( "[General Memory Manager] - IN.  ( void )." );
         auto handle = m_QueryScopeMemoryBinHandler.get_tail( );
         auto rootQueryScope = m_QueryScopeMemoryBinHandler.get_root( );
         while( handle != rootQueryScope ) {
            debug( "[General Memory Manager] - Freeing query scoped memory ", handle->m_BasePtr, " in Bin ", handle, "." );
            stdlib_free(handle->m_BasePtr);
            handle = handle->prev();
         }
         handle = m_GlobalScopeMemoryBinHandler.get_tail();
         auto rootGlobalScope = m_GlobalScopeMemoryBinHandler.get_root( );
         while (handle != rootGlobalScope) {
            debug( "[General Memory Manager] - Freeing global scoped memory ", handle->m_BasePtr, " in Bin ", handle, "." );
            stdlib_free(handle->m_BasePtr);
            handle = handle->prev();
         }
         trace( "[General Memory Manager] - OUT. ( void )." );
      }

   private:
      general_memory_manager(void) :
         abstract_memory_manager{},
         m_Initialized{(
                          (stdlib_malloc_ptr == nullptr) ||
                          (stdlib_malloc_ptr == nullptr) ||
                          (stdlib_malloc_ptr == nullptr)
                       ) ? init_mem_hooks() : true
         },
         m_GlobalScopeMemoryBinHandler(this),
         m_QueryScopeMemoryBinHandler(this) {
         trace( "[General Memory Manager] - IN.  ( void )." );
         numa_available();
         if (!m_Initialized)
            handle_error();
         trace(
            "[General Memory Manager] - OUT. ( this: ", this,
            ". Global Scoped Memory Bin Handler: ", &m_GlobalScopeMemoryBinHandler ,
            ". Query Scoped Memory Bin Handler: ", &m_QueryScopeMemoryBinHandler , " ).");
      }

   public:

      void *allocate(size_t p_AllocSize) override {
         trace( "[General Memory Manager] - IN.  ( AllocSize = ", p_AllocSize, " )." );
         size_t allocSize = get_size_with_alignment_padding(p_AllocSize);
         void * tmp = stdlib_malloc(allocSize);

         debug( "[General Memory Manager] - Allocated ", allocSize, " Bytes ( @ position: ", tmp, " ).");
         if(MSV_CXX_ATTRIBUTE_LIKELY(tmp != nullptr)) {
            debug( "[General Memory Manager] - Add space to global scoped Memory Bin Handler." );
            tmp = m_GlobalScopeMemoryBinHandler.append_bin(this, tmp, allocSize);
            trace( "[General Memory Manager] - OUT. ( Aligned Pointer: ", tmp, " )." );
            return tmp;
         } else {
            wtf( "[General Memory Manager] - allocate( AllocSize = ", allocSize, " ). Could not allocate the memory." );
            handle_error();
            trace( "[General Memory Manager] - OUT. ( nullptr )." );
            return nullptr;
         }
      }

      void *allocate(abstract_memory_manager *const p_Caller, size_t p_AllocSize) override {
         trace( "[General Memory Manager] - IN.  ( Caller = ", p_Caller, ". AllocSize = ", p_AllocSize, " )." );
         if (instanceof<general_memory_manager>(p_Caller)) {
            wtf( "[General Memory Manager] - Can not be called with static general memory manager as caller.");
            handle_error();
         }
         size_t allocSize = get_size_with_alignment_padding(p_AllocSize);
         void * tmp = stdlib_malloc( allocSize );
         debug( "[General Memory Manager] - Allocated ", allocSize, " Bytes ( @ position: ", tmp, " ).");
         if(MSV_CXX_ATTRIBUTE_LIKELY(tmp != nullptr)) {
            debug( "[General Memory Manager] - Add space to query scoped Memory Bin Handler." );
            tmp = m_QueryScopeMemoryBinHandler.append_bin(p_Caller, tmp, p_AllocSize);
            trace( "[General Memory Manager] - OUT. ( Aligned Pointer: ", tmp, " )." );
            return tmp;
         } else {
            wtf( "[General Memory Manager] - allocate( Caller = ", p_Caller, ". AllocSize = ", p_AllocSize, " ): Could not allocate ", p_AllocSize, " Bytes." );
            handle_error( );
            trace( "[General Memory Manager] - OUT. ( nullptr )." );
            return nullptr;
         }
      }

      void deallocate(MSV_CXX_ATTRIBUTE_PPUNUSED abstract_memory_manager *const p_Caller, MSV_CXX_ATTRIBUTE_PPUNUSED void *const p_Ptr ) override {
         trace( "[General Memory Manager] - IN.  ( Caller = ", p_Caller, ". Pointer = ", p_Ptr, " )." );
         warn( "[General Memory Manager] - Deallocate should not be invoked on the General Memory Manager." );
         // NOP
      }

      void deallocate(MSV_CXX_ATTRIBUTE_PPUNUSED void *const p_Ptr ) override {
         trace( "[General Memory Manager] - IN.  ( Pointer = ", p_Ptr, " )." );
         auto handle = m_GlobalScopeMemoryBinHandler.get_tail( );
         auto root = m_GlobalScopeMemoryBinHandler.get_root( );
         while(handle!= root) {
            trace(
               "[General Memory Manager] - Checking Handle ", handle,
               " ( base ptr = ", handle->m_BasePtr, ". aligned ptr = ",
               handle->m_AlignedPtr, ". size = ", handle->m_SizeByte, " Bytes ). ");
            if(handle->m_AlignedPtr == p_Ptr) {
               trace( "[General Memory Manager] - Remove handle and free space." );
               m_GlobalScopeMemoryBinHandler.remove_handle_and_free(handle);
               trace( "[General Memory Manager] - OUT. ( void )." );
               return;
            }
         }
         warn( "[General Memory Manager] - OUT. ( memory not found ).");
      }

      void * reallocate(void * p_Ptr, size_t p_AllocSize) override {
         auto handle = m_GlobalScopeMemoryBinHandler.get_tail( );
         auto root = m_GlobalScopeMemoryBinHandler.get_root( );
         while(handle != root) {
            if(handle->m_AlignedPtr == p_Ptr) {
               size_t allocSize = get_size_with_alignment_padding(p_AllocSize);
               return stdlib_realloc( handle->m_BasePtr, allocSize );
            }
            handle = handle->prev();
         }
         warn( "[General Memory Manager] - OUT. ( nullptr, because specified memory not found ).");
         return nullptr;
      }

      void * reallocate(abstract_memory_manager * const, void *, size_t) override {
         warn( "[Query Memory Manager] - Allocate from the context of a different Memory Manager should not be invoked on the Query Memory Manager." );
         return nullptr;
      }
      void handle_error(void) override {
         warn( "[General Memory Manager] - @TODO: Not implemented yet." );
         //@todo IMPLEMENT
         exit(1);
      }

      void destroy(abstract_memory_manager *const p_Caller) {
         trace( "[General Memory Manager] - IN.  ( Caller = ", p_Caller, " )." );
         if (instanceof<general_memory_manager>(p_Caller)) {
            wtf( "[General Memory Manager] - Can not be called with static general memory manager as caller.");
            handle_error();
         }
         auto handle = m_QueryScopeMemoryBinHandler.find_last(p_Caller);
         if (handle == m_QueryScopeMemoryBinHandler.get_tail()) {
            while (handle != nullptr) {
               trace("[General Memory Manager] - Freeing Memory ( ", handle->m_BasePtr, " ) from Bin ( ", handle, " )." );
               auto tmpHandle =
                  m_QueryScopeMemoryBinHandler.find_and_remove_reverse_until_first_other(p_Caller, handle);
               m_QueryScopeMemoryBinHandler.set_tail(tmpHandle);
               handle =
                  m_QueryScopeMemoryBinHandler.find_prev(
                     p_Caller,
                     tmpHandle);
            }
         } else {
            while (handle != nullptr) {
               trace("[General Memory Manager] - Freeing Bin from Caller ( ", handle->m_BasePtr, " ). ");
               handle =
                  m_QueryScopeMemoryBinHandler.find_prev(
                     p_Caller,
                     m_QueryScopeMemoryBinHandler.find_and_remove_reverse_until_first_other(p_Caller, handle)
                  );
            }
         }
         trace( "[General Memory Manager] - OUT. ( void )." );
      }

   private:
      bool m_Initialized;
      memory_bin_handler m_GlobalScopeMemoryBinHandler;
      memory_bin_handler m_QueryScopeMemoryBinHandler;

   };

}

#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_GENERAL_MM_H

