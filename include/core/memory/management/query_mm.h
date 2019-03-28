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
 * @file query_mm.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_QUERY_MM_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_QUERY_MM_H

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_ABSTRACT_MM_H
#  error "Abstract memory manager ( management/abstract_mm.h ) has to be included before general memory manager."
#endif

#ifndef MORPHSTORE_CORE_MEMORY_GLOBAL_MM_HOOKS_H
#  error "Memory Hooks ( global/mm_hooks.h ) has to be included before general memory manager."
#endif

#ifndef MSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE
#  define MSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE 128_MB
#endif

#include <core/utils/basic_types.h>
#include <core/utils/helper.h>
#include <core/utils/preprocessor.h>

#include <core/memory/management/utils/memory_bin_handler.h>

#include <cstring>

namespace morphstore {

class query_memory_manager_state_helper {
   private:
      bool m_Alive;
      query_memory_manager_state_helper( ) :
         m_Alive{ false } { }
      void set_alive( bool p_Alive ) {
         trace( "[Query Memory Manager State Helper] - IN.  ( ", p_Alive, " )." );
         trace( "[Query Memory Manager State Helper] - Current state: ", m_Alive, "." );
         m_Alive = p_Alive;
         trace( "[Query Memory Manager State Helper] - OUT. ( void )." );
      }
      friend class query_memory_manager;
   public:
      static query_memory_manager_state_helper & get_instance(void) {
         trace( "[Query Memory Manager State Helper] - IN.  ( void )." );
         static thread_local query_memory_manager_state_helper instance;
         trace( "[Query Memory Manager State Helper] - OUT. ( Instance: ", &instance, " )." );
         return instance;
      }
      bool is_alive(void) const {
         trace( "[Query Memory Manager State Helper] - IN.  ( void )." );
         trace( "[Query Memory Manager State Helper] - OUT. ( ", m_Alive, " )." );
         return m_Alive;
      }
};


class query_memory_manager : public abstract_memory_manager {
   public:
      static query_memory_manager & get_instance( size_t p_InitSpaceSize = 128_MB ) {
         trace( "[Query Memory Manager] - IN.  ( void )." );
         static thread_local query_memory_manager instance( p_InitSpaceSize );
         trace( "[Query Memory Manager] - OUT. ( Instance: ", &instance, " )." );
         return instance;
      }

      query_memory_manager( query_memory_manager const & ) = delete;
      void operator=( query_memory_manager const & ) = delete;
      virtual ~query_memory_manager( void ) {
         trace( "[Query Memory Manager] - IN.  ( this: ", this, " ).");
         m_GeneralMemoryManager.destroy( this );
         query_memory_manager_state_helper::get_instance().set_alive( false );
         trace( "[Query Memory Manager] - OUT.");
      }
   private:
      explicit query_memory_manager( size_t p_InitSpaceSize ) :
         abstract_memory_manager{},
         m_GeneralMemoryManager{ general_memory_manager::get_instance( ) },
         m_CurrentPtr{ nullptr },
         m_SpaceLeft{ 0 } {
         trace( "[Query Memory Manager] - IN.  ( Initial Size = ", ( p_InitSpaceSize / 1024 / 1024 ), " MB )." );
         void * tmp = m_GeneralMemoryManager.allocate( this, p_InitSpaceSize );
         if( tmp != nullptr ) {
            debug( "[Query Memory Manager] - Aquired ", p_InitSpaceSize, " Bytes ( @ position: ", tmp, " ).");
            m_CurrentPtr = tmp;
            m_SpaceLeft = p_InitSpaceSize;
            debug( "[Query Memory Manager] - Head = ", m_CurrentPtr, ". Space Lef = ", m_SpaceLeft, " Bytes." );
            query_memory_manager_state_helper::get_instance().set_alive( true );
         } else {
            wtf( "[Query Memory Manager] - query_memory_managerCould not aquire ", p_InitSpaceSize, " Bytes query scoped memory." );
            m_GeneralMemoryManager.handle_error( );
         }
         trace( "[Query Memory Manager] - OUT. ( this: ", this, " )." );
      }

   public:

      void * allocate( size_t p_AllocSize ) override {
         trace( "[Query Memory Manager] - IN.  ( AllocSize = ", p_AllocSize, " )." );

         //increasing the allocated size by one size_t, so the actual size can be stored at the very first item.
         size_t allocSize = get_size_with_alignment_padding(p_AllocSize) + sizeof(size_t);
         trace( "[Query Memory Manager] - Have to allocate ", allocSize, " Bytes ).");
         void * tmp = m_CurrentPtr;
         debug( "[Query Memory Manager] - head = ", m_CurrentPtr, ". Space Left = ", m_SpaceLeft, " Bytes." );
         if(MSV_CXX_ATTRIBUTE_UNLIKELY(m_SpaceLeft < allocSize)) {
            debug(
               "[Query Memory Manager] - No Space Left. ( Needed: ", allocSize,
               " Bytes. Available: ", m_SpaceLeft, " Bytes )." );
            size_t nextExpandSize = expander.next_size( allocSize );
            trace( "[Query Memory Manager] - Requesting ", nextExpandSize, " Bytes from global scoped memory." );
            tmp = m_GeneralMemoryManager.allocate( this, nextExpandSize );
            m_SpaceLeft = nextExpandSize;
            trace( "[Query Memory Manager] - New head = ", m_CurrentPtr, ". Space Left = ", m_SpaceLeft, " Bytes." );
         }

         if(MSV_CXX_ATTRIBUTE_LIKELY(tmp != nullptr)) {
            trace( "[Query Memory Manager] - Creating extended aligned ptr (current = ", tmp, ". Size = ", p_AllocSize, " Bytes." );
            void * result_ptr = create_extended_aligned_ptr(tmp, p_AllocSize);
            size_t bytes_lost = reinterpret_cast<size_t>(tmp)-reinterpret_cast<size_t>(m_CurrentPtr);
            trace(
               "[Query Memory Manager] - Set new Head to aligned ptr ( ", result_ptr, " ). ",
               "Bytes lost through Alignment: ",
               bytes_lost, ".");
            m_CurrentPtr = static_cast< char * >( result_ptr ) + p_AllocSize;
            m_SpaceLeft -= ( p_AllocSize + bytes_lost );
            trace( "[Query Memory Manager] - OUT. ( pointer: ", result_ptr, ". head = ", m_CurrentPtr, ". Space Left = ", m_SpaceLeft, " Bytes)." );
            return result_ptr;
         } else {
            m_GeneralMemoryManager.handle_error( );
            wtf( "[Query Memory Manager] - Could not aquire ", allocSize, " Bytes query scoped memory." );
            return nullptr;
         }
      }

      void * allocate( MSV_CXX_ATTRIBUTE_PPUNUSED abstract_memory_manager * const p_Caller, MSV_CXX_ATTRIBUTE_PPUNUSED size_t p_AllocSize ) override {
         warn( "[Query Memory Manager] - Allocate from the context of a different Memory Manager should not be invoked on the Query Memory Manager." );
         return nullptr;
      }

      void deallocate( MSV_CXX_ATTRIBUTE_PPUNUSED abstract_memory_manager * const p_Caller, MSV_CXX_ATTRIBUTE_PPUNUSED void * const p_Ptr ) override {
         trace( "[Query Memory Manager] - IN.  ( Caller = ", p_Caller, ". Pointer = ", p_Ptr, " )." );
         info( "[Query Memory Manager] - Deallocate ( abstract_memory_manager * const, void * const ) ",
               "should not be invoked on the Query Memory Manager." );
      }

      void deallocate( MSV_CXX_ATTRIBUTE_PPUNUSED void * const p_Ptr ) override {
         trace( "[Query Memory Manager] - IN.  ( Pointer = ", p_Ptr, " )." );
         info( "[Query Memory Manager] - Deallocate ( void * const ) should not be invoked on the Query Memory Manager." );
      }


      void * reallocate(void * p_Ptr, size_t p_AllocSize) override {
         if(MSV_CXX_ATTRIBUTE_UNLIKELY(p_Ptr == nullptr))
            return allocate(p_AllocSize);
         if(MSV_CXX_ATTRIBUTE_UNLIKELY(p_AllocSize == 0))
            return nullptr;
         void * result = allocate(p_AllocSize);
         if(MSV_CXX_ATTRIBUTE_UNLIKELY(result == nullptr))
            return nullptr;
         std::memcpy(result, p_Ptr, (reinterpret_cast<size_t *>(p_Ptr))[-1]);
         return result;
      }

      void * reallocate(abstract_memory_manager * const, void *, size_t) override {
         warn( "[Query Memory Manager] - Allocate from the context of a different Memory Manager should not be invoked on the Query Memory Manager." );
         return nullptr;
      }


      void handle_error( void ) override {
      }




   private:
      general_memory_manager & m_GeneralMemoryManager;
      void * m_CurrentPtr;
      size_t m_SpaceLeft;
      mm_expand_strategy_chunk_based< MSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE > expander;

};

}

#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_QUERY_MM_H
