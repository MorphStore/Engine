/**
 * @file mm_impl.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MM_IMPL_H
#define MORPHSTORE_CORE_MEMORY_MM_IMPL_H

#include "../utils/types.h"
#include "../utils/preprocessor.h"
#include "mm_hooks.h"
#include "mm_helper.h"

#include <cstdlib>
#include <cstdio>

#ifndef QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE
#define QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE 128_MB
#endif


namespace morphstore { namespace memory {

class general_memory_manager : public abstract_memory_manager {
   public:
      static general_memory_manager & get_instance( void ) {
         static general_memory_manager instance;
         return instance;
      }
      general_memory_manager( general_memory_manager const & ) = delete;
      void operator=( general_memory_manager const & ) = delete;
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

      void * allocate( PPUNUSED size_t p_AllocSize ) override {
         //LOG: THIS SHOULD NOT BE CALLED!
         return nullptr;
      }
      void * allocate( abstract_memory_manager * p_Caller, size_t p_AllocSize ) override {
         void * tmp = stdlib_malloc( p_AllocSize );
         if( tmp != nullptr ) {
            m_MemoryBinHandler.append_bin( p_Caller, tmp, p_AllocSize );
            return tmp;
         } else {
            fprintf( stderr, "GMM[alloc]: Could not allocate %zu Bytes.\n", p_AllocSize );
            handle_error( );
            return nullptr;
         }
      }

      void deallocate( PPUNUSED abstract_memory_manager * p_Caller, PPUNUSED void * ) override {
         // NOP
      }

      void deallocate( PPUNUSED void * )  override {
         // NOP
      }

      void handle_error( void ) override {
         //@todo IMPLEMENT
         exit( 1 );
      }
      void destroy( abstract_memory_manager * p_Caller ) {
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
            fprintf( stderr, "GMM[alloc]: Could not persist-allocate %zu Bytes.\n", p_AllocSize );
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
            fprintf( stderr, "QMM[ctor]: Could not allocate %zu Bytes.\n", p_InitSpaceSize );
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
            fprintf( stderr, "QMM[alloc]: Could not allocate %zu Bytes.\n", p_AllocSize );
            m_GeneralMemoryManager.handle_error( );
         }
         return tmp;
      }

      void * allocate( PPUNUSED abstract_memory_manager * p_Caller, PPUNUSED size_t p_AllocSize ) override {
         //LOG: THIS SHOULD NOT BE CALLED!
         return nullptr;
      }

      void deallocate( PPUNUSED abstract_memory_manager * p_Caller, PPUNUSED void * ) override {
         // NOP
      }

      void deallocate( PPUNUSED void * ) override {
         // NOP
      }

      void handle_error( void ) override {
         //LOG: THIS SHOULD NOT BE CALLED!
      }


   private:
      general_memory_manager & m_GeneralMemoryManager;
      void * m_CurrentPtr;
      size_t m_SpaceLeft;
      mm_expand_strategy_chunk_based< QUERY_MEMORY_MANAGER_MINIMUM_EXPAND_SIZE > expander;

};

} }
#endif //MORPHSTORE_CORE_MEMORY_MM_IMPL_H
