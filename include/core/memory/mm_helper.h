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
 * @file mm_helper.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MM_HELPER_H
#define MORPHSTORE_CORE_MEMORY_MM_HELPER_H

#include "../utils/types.h"
#include "../utils/math.h"
#include "mm_hooks.h"
#include "mm.h"

#include <cassert>
#include <cstdio>


namespace morphstore { namespace memory {






template< size_t MinChunkSize >
constexpr size_t chunk_size( size_t pRequestedSize ) {
   static_assert(
      is_power_of_two( MinChunkSize ),
      "For performance and convenience granularity (pN) has to be a power of 2.");
   if( MinChunkSize >= pRequestedSize )
      return MinChunkSize;
   size_t remainder = pRequestedSize & ( MinChunkSize - 1 );
   return ( remainder == 0 ) ? pRequestedSize : pRequestedSize + ( MinChunkSize - remainder );
}


size_t chunk_size( size_t MinChunkSize, size_t pRequestedSize ) {
   assert( is_power_of_two( MinChunkSize ) );
   if( MinChunkSize >= pRequestedSize )
      return MinChunkSize;
   size_t remainder = pRequestedSize & ( MinChunkSize - 1 );
   return ( remainder == 0 ) ? pRequestedSize : pRequestedSize + ( MinChunkSize - remainder );
}


class mm_expand_strategy {
   protected:
      size_t mCurrentSize;
   public:
      constexpr mm_expand_strategy( void ) : mCurrentSize{ 0 } { }
      constexpr mm_expand_strategy( size_t pCurrentSize ) : mCurrentSize{ pCurrentSize } { }
};

template< size_t MinimumExpandSize >
class mm_expand_strategy_chunk_based : public mm_expand_strategy {
      static_assert( is_power_of_two( MinimumExpandSize ),
                     "For performance and convenience granularity (pN) has to be a power of 2." );
   public:
      constexpr mm_expand_strategy_chunk_based( ) : mm_expand_strategy( MinimumExpandSize ) { }

      inline size_t current_size( void ) const {
         return mCurrentSize;
      }

      inline size_t next_size( size_t pExpandSize ) {
         mCurrentSize = chunk_size< MinimumExpandSize >( pExpandSize );
         return mCurrentSize;
      }
};
template< size_t MinimumExpandSize >
class mm_expand_strategy_chunk_based_quadratic : public mm_expand_strategy {
      static_assert( is_power_of_two( MinimumExpandSize ),
                     "For performance and convenience granularity (pN) has to be a power of 2." );
   public:
      constexpr mm_expand_strategy_chunk_based_quadratic( ) : mm_expand_strategy( MinimumExpandSize ) { }

      inline size_t current_size( void ) const {
         return mCurrentSize;
      }

      inline size_t next_size( size_t pExpandSize ) {
         mCurrentSize = chunk_size( mCurrentSize << 1, pExpandSize );
         return mCurrentSize;
      }
};


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

         memory_bin_handle( memory_bin_handle const &  ) = delete;
         memory_bin_handle( memory_bin_handle && ) = delete;
         memory_bin_handle & operator=( memory_bin_handle const & ) = default;
         memory_bin_handle & operator=( memory_bin_handle && ) = default;
         memory_bin_handle( abstract_memory_manager * p_MemoryManager, void * p_BasePtr, size_t p_SizeByte ) :
            m_MemoryManager{ p_MemoryManager },
            m_BasePtr{ p_BasePtr },
            m_SizeByte{ p_SizeByte },
            m_PrevHandle{ nullptr },
            m_NextHandle{ nullptr }{
            debug(
               "Memory Bin Handle - ctor( Owner =", p_MemoryManager,
               ", Data =", p_BasePtr,
               ", Size =", p_SizeByte,
               ") this =", this
            );
         }
         memory_bin_handle( abstract_memory_manager * p_MemoryManager, void * p_BasePtr, size_t p_SizeByte, memory_bin_handle * p_PrevHandle ) :
            m_MemoryManager{ p_MemoryManager },
            m_BasePtr{ p_BasePtr },
            m_SizeByte{ p_SizeByte },
            m_PrevHandle{ p_PrevHandle },
            m_NextHandle{ nullptr }{
            debug(
               "Memory Bin Handle - ctor( Owner =", p_MemoryManager,
               ", Data =", p_BasePtr, ", Size =", p_SizeByte,
               ", PrevHandle =", p_PrevHandle,
               ") this =", this
            );
         }
         memory_bin_handle( abstract_memory_manager * p_MemoryManager, void * p_BasePtr, size_t p_SizeByte, memory_bin_handle * p_PrevHandle, memory_bin_handle * p_NextHandle ) :
            m_MemoryManager{ p_MemoryManager },
            m_BasePtr{ p_BasePtr },
            m_SizeByte{ p_SizeByte },
            m_PrevHandle{ p_PrevHandle },
            m_NextHandle{ p_NextHandle }{
            debug(
               "Memory Bin Handle - ctor( Owner =", p_MemoryManager,
               ", Data =", p_BasePtr, ", Size =", p_SizeByte,
               ", PrevHandle =", p_PrevHandle,
               ", NextHandle =", p_NextHandle,
               ") this =", this
            );
         }
         inline void init(  abstract_memory_manager * p_MemoryManager, void * p_BasePtr, size_t p_SizeByte, memory_bin_handle * p_PrevHandle, memory_bin_handle * p_NextHandle  ) {
            m_MemoryManager = p_MemoryManager;
            m_BasePtr = p_BasePtr;
            m_SizeByte = p_SizeByte;
            m_PrevHandle = p_PrevHandle;
            m_NextHandle = p_NextHandle;
            debug(
               "Memory Bin Handle - init( Owner =", p_MemoryManager,
               ", Data =", p_BasePtr, ", Size =", p_SizeByte,
               ", PrevHandle =", p_PrevHandle,
               ", NextHandle =", p_NextHandle,
               ") this =", this
            );
         }
         void remove( void ) {
            if( m_PrevHandle != nullptr ) {
               if( m_NextHandle != nullptr ) {
                  m_PrevHandle->m_NextHandle = m_NextHandle;
                  m_NextHandle->m_PrevHandle = m_PrevHandle;
               } else {
                  m_PrevHandle->m_NextHandle = nullptr;
               }
            } else {
               if( m_NextHandle != nullptr )
                  m_NextHandle->m_PrevHandle = nullptr;
            }
         }
         inline memory_bin_handle * next( ) const {
            return m_NextHandle;
         }
      };
   public:
      memory_bin_handler( abstract_memory_manager * const p_MemoryManager, void * const p_BasePtr, size_t p_SizeByte ) :
         m_BinHandleStructRoot{ nullptr }, m_BinHandleStructTail{ nullptr }{
         memory_bin_handle * tmp = static_cast< memory_bin_handle * >( stdlib_malloc( sizeof( memory_bin_handle ) ) );
         if( tmp != nullptr ) {
            tmp->init( p_MemoryManager, p_BasePtr, p_SizeByte, nullptr, nullptr );
            m_BinHandleStructRoot = tmp;
            m_BinHandleStructTail = tmp;
         } else {
            wtf( "Memory Bin Handler - ctor(): Could not allocate ", sizeof( memory_bin_handle ), " Bytes for a handle." );
            p_MemoryManager->handle_error( );
         }
         debug(
            "Memory Bin Handler - ctor( Associated with", p_MemoryManager,
            ", Data =", p_BasePtr,
            ", Size =", p_SizeByte,
            ") this =", this
         );
      }
      memory_bin_handler( void ) : m_BinHandleStructRoot{ nullptr }, m_BinHandleStructTail{ nullptr }{
         debug(
            "Memory Bin Handler - ctor( Associated with", 0,
            ", Data = 0",
            ", Size = 0",
            ") this =", this
         );
      }
      ~memory_bin_handler( void ){
         debug( "Memory Bin Handler - dtor( )" );
         memory_bin_handle * handle = m_BinHandleStructRoot;
         memory_bin_handle * next_handle = nullptr;
         while( handle != nullptr ) {
            next_handle = handle->m_NextHandle;
            debug( "Memory Bin Handler - dtor( ). Freeing", handle );
            stdlib_free( static_cast< void * >( handle ) );
            handle = next_handle;
         }

      }
   private:
      memory_bin_handle * m_BinHandleStructRoot;
      memory_bin_handle * m_BinHandleStructTail;

   public:

      inline void append_bin( abstract_memory_manager * const p_MemoryManager, void * const p_BasePtr, size_t p_BinSize ) {
         debug(
            "Memory Bin Handler - append_bin( Associated with", p_MemoryManager,
            ", Data =", p_BasePtr,
            ", Size =", p_BinSize,
            ")"
         );
         memory_bin_handle * tmp = static_cast< memory_bin_handle * >( stdlib_malloc( sizeof( memory_bin_handle ) ) );
         if( tmp != nullptr ) {
            tmp->init( p_MemoryManager, p_BasePtr, p_BinSize, m_BinHandleStructTail, nullptr );
            if( m_BinHandleStructRoot == nullptr ) {
               m_BinHandleStructRoot = tmp;
               m_BinHandleStructTail = tmp;
//            if( m_BinHandleStructTail == nullptr ) {
//               m_BinHandleStructRoot = m_BinHandleStructTail = tmp;
//            } else {
//               m_BinHandleStructTail->m_NextHandle = tmp;
//               m_BinHandleStructTail = tmp;
            }
         } else {
            wtf( "Memory Bin Handler - append_bin(): Could not allocate ", sizeof( memory_bin_handle ), " Bytes for a handle." );
            p_MemoryManager->handle_error( );
         }
         debug(
            "Memory Bin Handler - append_bin( ) -",
            "Prev Handle =", tmp->m_PrevHandle,
            "Next Handle =", tmp->m_NextHandle,
            "Appended. Root =", m_BinHandleStructRoot, "Tail =", m_BinHandleStructTail
         );

      }
      inline memory_bin_handle * remove_bin( memory_bin_handle * const handle ) {
         memory_bin_handle * next = handle->m_NextHandle;
         handle->remove();
         if( handle == m_BinHandleStructRoot )
            m_BinHandleStructRoot = handle->m_NextHandle;
         if( handle == m_BinHandleStructTail )
            m_BinHandleStructTail = handle->m_PrevHandle;
         stdlib_free( static_cast< void * >( handle ) );
         return next;
      }

      inline memory_bin_handle * find_first( abstract_memory_manager * const p_MemoryManager ) const {
         memory_bin_handle * handle = m_BinHandleStructRoot;
         while( handle != nullptr ) {
            if( handle->m_MemoryManager == p_MemoryManager )
               return handle;
            handle = handle->m_NextHandle;
         }
         return handle;
      }

      inline memory_bin_handle * find_next( abstract_memory_manager * const p_MemoryManager, memory_bin_handle * p_CurrentHandle ) const {
         memory_bin_handle * handle = p_CurrentHandle;
         while( handle != nullptr ) {
            if( handle->m_MemoryManager == p_MemoryManager )
               return handle;
            handle = handle->m_NextHandle;
         }
         return nullptr;
      }

      inline memory_bin_handle * get_root( void ) const {
         return m_BinHandleStructRoot;
      }
      inline memory_bin_handle * get_tail( void ) const {
         return m_BinHandleStructTail;
      }
};



} }


#endif //MORPHSTORE_CORE_MEMORY_MM_HELPER_H
