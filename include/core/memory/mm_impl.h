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
 * @file mm_impl.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MM_IMPL_H
#define MORPHSTORE_CORE_MEMORY_MM_IMPL_H

#ifndef MORPHSTORE_CORE_MEMORY_MM_GLOB_H
#  error "mm_impl.h has to be included AFTER mm_glob.h"
#endif
#ifndef MSV_NO_SELFMANAGED_MEMORY

#include "../utils/types.h"
#include "../utils/preprocessor.h"
#include "mm_hooks.h"
#include "mm_helper.h"
#include "../utils/logger.h"

#include <cstdlib>
#include <cstdio>

#ifndef MSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE
#define MSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE 128_MB
#endif


namespace morphstore { namespace memory {

class general_memory_manager : public abstract_memory_manager {
   public:
      static general_memory_manager & get_instance( void ) {
         static general_memory_manager instance;
         return instance;
      }
      general_memory_manager( general_memory_manager const & ) = delete;
      general_memory_manager & operator=( general_memory_manager const & ) = delete;
      ~general_memory_manager( void ) {
         auto * handle = m_MemoryBinHandler.get_root( );
         while( handle != nullptr ) {
            stdlib_free( handle->m_BasePtr );
            handle->next();
         }
      }
   private:
      general_memory_manager( void ) = default;

   public:

      void * allocate( MSV_PPUNUSED size_t p_AllocSize ) override {
         warn( "General Memory Manager - allocate(): Called allocate() without abstract_memory_manager directly on the GMM. This should not happed!" );
         return nullptr;
      }
      void * allocate( abstract_memory_manager * const p_Caller, size_t p_AllocSize ) override {
         void * tmp = stdlib_malloc( p_AllocSize );
         if( tmp != nullptr ) {
            m_MemoryBinHandler.append_bin( p_Caller, tmp, p_AllocSize );
            return tmp;
         } else {
            wtf( "General Memory Manager - allocate(): Could not allocate ", p_AllocSize, " Bytes." );
            handle_error( );
            return nullptr;
         }
      }

      void deallocate( MSV_PPUNUSED abstract_memory_manager * const, MSV_PPUNUSED void * const ) override {
         // NOP
      }

      void deallocate( MSV_PPUNUSED void * const )  override {
         // NOP
      }

      void handle_error( void ) override {
         //@todo IMPLEMENT
         exit( 1 );
      }
      void destroy( abstract_memory_manager * const p_Caller ) {
         auto handle = m_MemoryBinHandler.find_first( p_Caller );
         while( handle != nullptr ) {
            stdlib_free( handle->m_BasePtr );
            handle = m_MemoryBinHandler.find_next( p_Caller, m_MemoryBinHandler.remove_bin( handle ) );
         }
      }
      void * allocate_persist( size_t p_AllocSize ) {
         void * tmp = stdlib_malloc( p_AllocSize );
         if( tmp != nullptr ) {
            m_MemoryBinHandler.append_bin( this, tmp, p_AllocSize );
            return tmp;
         } else {
            wtf( "General Memory Manager - allocate_persist(): Could not allocate ", p_AllocSize, " Bytes." );
            handle_error( );
            return nullptr;
         }
      }

   private:
      memory_bin_handler m_MemoryBinHandler;
};

class query_memory_manager : public abstract_memory_manager {
   public:
      static query_memory_manager & get_instance( size_t p_InintSpaceSize = 128_MB ) {
         static thread_local query_memory_manager instance( p_InintSpaceSize );
         return instance;
      }

      query_memory_manager( query_memory_manager const & ) = delete;
      void operator=( query_memory_manager const & ) = delete;
      ~query_memory_manager( void ) {
         m_GeneralMemoryManager.destroy( this );
      }
   private:
      explicit query_memory_manager( size_t p_InitSpaceSize ) :
         m_GeneralMemoryManager{ general_memory_manager::get_instance( ) },
         m_CurrentPtr{ nullptr },
         m_SpaceLeft{ 0 } {
         void * tmp = m_GeneralMemoryManager.allocate( this, p_InitSpaceSize );
         if( tmp != nullptr ) {
            m_CurrentPtr = tmp;
            m_SpaceLeft = p_InitSpaceSize;
         } else {
            wtf( "Query Memory Manager - ctor(): Could not allocate ", p_InitSpaceSize, " Bytes." );
            m_GeneralMemoryManager.handle_error( );
         }
      }

   public:

      void * allocate( size_t p_AllocSize ) override {
         if( m_SpaceLeft < p_AllocSize ) {
            size_t nextExpandSize = expander.next_size( p_AllocSize );
            m_CurrentPtr = m_GeneralMemoryManager.allocate( this, nextExpandSize );
            m_SpaceLeft = nextExpandSize;
         }
         void * tmp = m_CurrentPtr;
         if( m_CurrentPtr != nullptr ) {
            m_CurrentPtr = static_cast< char * >( m_CurrentPtr ) + p_AllocSize;
            m_SpaceLeft -= p_AllocSize;
         } else {
            wtf( "Query Memory Manager - allocate(): Could not allocate ", p_AllocSize, " Bytes." );
            m_GeneralMemoryManager.handle_error( );
         }
         return tmp;
      }

      void * allocate( MSV_PPUNUSED abstract_memory_manager * const p_Caller, MSV_PPUNUSED size_t p_AllocSize ) override {
         warn( "Query Memory Manager - allocate(): Called allocate() with abstract_memory_manager. This is not intended to happen!" );
         return nullptr;
      }

      void deallocate( MSV_PPUNUSED abstract_memory_manager * const, MSV_PPUNUSED void * const ) override {
         // NOP
      }

      void deallocate( MSV_PPUNUSED void * const ) override {
         // NOP
      }

      void handle_error( void ) override {
         warn( "Query Memory Manager - handle_error(): This is not intended to happen!" );
      }


   private:
      general_memory_manager & m_GeneralMemoryManager;
      void * m_CurrentPtr;
      size_t m_SpaceLeft;
      mm_expand_strategy_chunk_based< MSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE > expander;

};

} }

#endif
#endif //MORPHSTORE_CORE_MEMORY_MM_IMPL_H
