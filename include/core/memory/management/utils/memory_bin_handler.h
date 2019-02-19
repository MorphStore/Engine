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
 * @file memory_bin_handler.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_UTILS_MEMORY_BIN_HANDLER_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_UTILS_MEMORY_BIN_HANDLER_H

#include <core/utils/logger.h>

namespace morphstore {

class memory_bin_handler {
   private:
      /**
      * @brief Struct for holding memory regions alongside the associated memory manager.
      *
      * @details Uses a Linked List for the memory bins.
      */
      struct memory_bin_handle {
         abstract_memory_manager * m_MemoryManager;
         void * m_BasePtr;
         size_t m_SizeByte;
         memory_bin_handle * m_PrevHandle;
         memory_bin_handle * m_NextHandle;

         memory_bin_handle( void ) = delete;
         memory_bin_handle( memory_bin_handle const &  ) = delete;
         memory_bin_handle( memory_bin_handle && ) = delete;
         memory_bin_handle & operator=( memory_bin_handle const & ) = default;
         memory_bin_handle & operator=( memory_bin_handle && ) = default;

         inline void init(  abstract_memory_manager * p_MemoryManager, void * p_BasePtr, size_t p_SizeByte, memory_bin_handle * p_PrevHandle, memory_bin_handle * p_NextHandle  ) {
            m_MemoryManager = p_MemoryManager;
            m_BasePtr = p_BasePtr;
            m_SizeByte = p_SizeByte;
            m_PrevHandle = p_PrevHandle;
            m_NextHandle = p_NextHandle;
            trace(
               "[Memory Bin Handle] - IN.  ( Owner = ", p_MemoryManager,
               ", Data = ", p_BasePtr, ", Size = ", p_SizeByte,
               ", PrevHandle = ", p_PrevHandle,
               ", NextHandle = ", p_NextHandle,
               " )." );
            trace( "[Memory Bin Handle] - OUT. ( void )" );
         }
         inline memory_bin_handle * prev( ) const {
            return m_PrevHandle;
         }
      };
   public:
      explicit memory_bin_handler( abstract_memory_manager * const p_MemoryManager ) {
         trace( "[Memory Bin Handler] - IN.  ( Owner = ", p_MemoryManager, " )." );
         if ( stdlib_malloc_ptr == nullptr ) {
            trace( "[Memory Bin Handler] - Initialize memory hooks." );
            if( ! init_mem_hooks( ) ) {
               wtf( "[Memory Bin Handler] - Could not initialize memory hooks" );
               p_MemoryManager->handle_error( );
            }
         }
         trace( "[Memory Bin Handler] - Create new Memory Bin handle." );
         memory_bin_handle * tmp = static_cast< memory_bin_handle * >( stdlib_malloc( sizeof( memory_bin_handle ) ) );
         if( tmp != nullptr ) {
            trace( "[Memory Bin Handler] - Initialize Bin Handle." );
            tmp->init( p_MemoryManager, nullptr, 0, nullptr, nullptr );
            trace( "[Memory Bin Handler] - Assign newly created handle ", tmp, " as root and tail." );
            m_BinHandleStructRoot = tmp;
            m_BinHandleStructTail = tmp;
         } else {
            wtf( "[Memory Bin Handler] - Could not create a Memory Bin Handle." );
            p_MemoryManager->handle_error( );
         }
         trace(
            "[Memory Bin Handler] - Root = ", m_BinHandleStructRoot, ". Tail = ", m_BinHandleStructTail, "."
         );
         trace( "[Memory Bin Handler] - OUT. ( this = ", this, " )" );
      }

      ~memory_bin_handler( void ){
         trace( "[Memory Bin Handler] - IN. ( void )." );
         memory_bin_handle * handle = m_BinHandleStructTail;
         memory_bin_handle * prev_handle = nullptr;
         while( handle != nullptr ) {
            prev_handle = handle->m_PrevHandle;
            trace( "[Memory Bin Handler] - Freeing Handle ( ", handle, " )." );
            stdlib_free( static_cast< void * >( handle ) );
            handle = prev_handle;
         }
         trace( "[Memory Bin Handler] - OUT. ( void ) " );

      }
   private:
      memory_bin_handle * m_BinHandleStructRoot;
      memory_bin_handle * m_BinHandleStructTail;

   public:

      inline void append_bin( abstract_memory_manager * const p_MemoryManager, void * const p_BasePtr, size_t p_BinSize ) {
         trace(
            "[Memory Bin Handler] - IN.  ( Owner = ", p_MemoryManager,
            ". Memory = ", p_BasePtr,
            ". Size = ", p_BinSize, " )."
         );
         trace( "[Memory Bin Handler] - Create new Memory Bin handle." );
         memory_bin_handle * tmp = static_cast< memory_bin_handle * >( stdlib_malloc( sizeof( memory_bin_handle ) ) );
         if( tmp != nullptr ) {
            trace( "[Memory Bin Handler] - Initialize Bin Handle." );
            tmp->init( p_MemoryManager, p_BasePtr, p_BinSize, m_BinHandleStructTail, nullptr );
            trace(
               "[Memory Bin Handler] - Assign newly created handle ", tmp, " as new tail. Reset pointer from old tail."
            );
            m_BinHandleStructTail->m_NextHandle = tmp;
            m_BinHandleStructTail = tmp;
         } else {
            wtf( "[Memory Bin Handler] - Could not allocate ", sizeof( memory_bin_handle ), " Bytes for a handle." );
            p_MemoryManager->handle_error( );
         }
         trace(
            "[Memory Bin Handler] - Root = ", m_BinHandleStructRoot, ". Tail = ", m_BinHandleStructTail, "."
         );
         trace( "[Memory Bin Handler] - OUT. ( void ).");
      }

      inline memory_bin_handle * get_tail( void ) const {
         return m_BinHandleStructTail;
      }
      inline memory_bin_handle * get_root( void ) const {
         return m_BinHandleStructRoot;
      }
      inline void set_tail( memory_bin_handle * const handle ) {
         m_BinHandleStructTail = handle;
      }
      inline memory_bin_handle * find_last( abstract_memory_manager * const p_MemoryManager ) const {
         trace( "[Memory Bin Handler] - IN.  ( Owner = ", p_MemoryManager, " )." );
         memory_bin_handle * handle = m_BinHandleStructTail;
         while( handle != m_BinHandleStructRoot ) {
            if( handle->m_MemoryManager == p_MemoryManager ) {
               trace( "[Memory Bin Handler] - OUT. ( ", handle, " )." );
               return handle;
            }
            handle = handle->m_PrevHandle;
         }
         trace( "[Memory Bin Handler] - OUT. No handle found. ( nullptr )." );
         return nullptr;
      }

      inline memory_bin_handle * find_prev(
         abstract_memory_manager * const p_MemoryManager,
         memory_bin_handle * const p_Current ) const {
         trace( "[Memory Bin Handler] - IN.  ( Owner = ", p_MemoryManager, ", Current Handle = ", p_Current, " )." );
         memory_bin_handle * handle = p_Current;
         while( handle != m_BinHandleStructRoot ) {
            if( handle->m_MemoryManager == p_MemoryManager ) {
               trace( "[Memory Bin Handler] - OUT. ( ", handle, " )." );
               return handle;
            }
            handle = handle->m_PrevHandle;
         }
         trace( "[Memory Bin Handler] - OUT. No handle found. ( nullptr )." );
         return nullptr;
      }

      inline memory_bin_handle * find_and_remove_reverse_until_first_other(
         abstract_memory_manager * const p_MemoryManager,
         memory_bin_handle * p_Current
      ) {

         info( "[Memory Bin Handler] - IN.  ( Owner = ", p_MemoryManager, ", Current Handle = ", p_Current, " )." );
         memory_bin_handle * nextHandle = p_Current->m_NextHandle;
         memory_bin_handle * handle = p_Current;
         memory_bin_handle * prevHandle;
         // we assume, that find_reverse_first_not is NOT called with p_MemoryManager == General_memory_manager
         // thus this loop terminates at least when it comes to root.
         while( handle->m_MemoryManager == p_MemoryManager ) {
            trace( "[Memory Bin Handler] - Freeing ", handle->m_BasePtr, " from Handle ", handle, "." );
            stdlib_free( static_cast< void * >( handle->m_BasePtr ) );
            prevHandle = handle->m_PrevHandle;
            trace( "[Memory Bin Handler] - Next: ", nextHandle, ". Prev: ", prevHandle, "." );
            trace( "[Memory Bin Handler] - Freeing handle ", handle, "." );
            stdlib_free( static_cast< void * >( handle ) );
            handle = prevHandle;
            trace( "[Memory Bin Handler] - Current handle: ", handle, "." );
         }
         handle->m_NextHandle = nextHandle;
         trace(
            "[Memory Bin Handler] - OUT. ( Handle not associated with ", p_MemoryManager,
            ": ", handle,
            " [ Prev: ", handle->m_PrevHandle,
            " . Next: ", handle->m_NextHandle,
            " . Memory: ", handle->m_BasePtr,
            " . Size: ", handle->m_SizeByte, " Bytes",
            " )." );
         return handle;
      }
};

}
#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_UTILS_MEMORY_BIN_HANDLER_H
