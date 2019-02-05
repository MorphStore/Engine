/**
 * @file query_mm.h
 * @brief Brief description
 * @author Johannes Pietrzyk
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

#include "../../utils/basic_types.h"
#include "../../utils/helper.h"
#include "../../utils/preprocessor.h"

#include "utils/memory_bin_handler.h"



namespace morphstore { namespace memory {

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
            m_GeneralMemoryManager.handle_error( );
         }
         return tmp;
      }

      void * allocate( MSV_PPUNUSED abstract_memory_manager * const p_Caller, MSV_PPUNUSED size_t p_AllocSize ) override {
         return nullptr;
      }

      void deallocate( MSV_PPUNUSED abstract_memory_manager * const, MSV_PPUNUSED void * const ) override {
      }

      void deallocate( MSV_PPUNUSED void * const ) override {
      }

      void handle_error( void ) override {
      }


   private:
      general_memory_manager & m_GeneralMemoryManager;
      void * m_CurrentPtr;
      size_t m_SpaceLeft;
      mm_expand_strategy_chunk_based< MSV_QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE > expander;

};

}}

#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_QUERY_MM_H
