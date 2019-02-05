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
      static query_memory_manager & get_instance( size_t p_InitSpaceSize = 128_MB ) {
         trace( "[Query Memory Manager] - IN.  ( void )." );
         static thread_local query_memory_manager instance( p_InitSpaceSize );
         trace( "[Query Memory Manager] - OUT. ( Instance: ", &instance, " )." );
         return instance;
      }

      query_memory_manager( query_memory_manager const & ) = delete;
      void operator=( query_memory_manager const & ) = delete;
      ~query_memory_manager( void ) {
         trace( "[Query Memory Manager] - IN.  ( this: ", this, " ).");
         m_GeneralMemoryManager.destroy( this );
      }
   private:
      explicit query_memory_manager( size_t p_InitSpaceSize ) :
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
         } else {
            wtf( "[Query Memory Manager] - Could not aquire ", p_InitSpaceSize, " Bytes ephimeral memory." );
            m_GeneralMemoryManager.handle_error( );
         }
         trace( "[Query Memory Manager] - OUT. ( this: ", this, " )." );
      }

   public:

      void * allocate( size_t p_AllocSize ) override {
         trace( "[Query Memory Manager] - IN.  ( AllocSize = ", p_AllocSize, " )." );
         size_t nextExpandSize = p_AllocSize;
         if( m_SpaceLeft < p_AllocSize ) {
            debug(
               "[Query Memory Manager] - No Space Left. ( Needed: ", p_AllocSize,
               " Bytes. Available: ", m_SpaceLeft, " Bytes )." );
            nextExpandSize = expander.next_size( p_AllocSize );
            trace( "[Query Memory Manager] - Requesting ", nextExpandSize, " Bytes from perpetual memory." );
            m_CurrentPtr = m_GeneralMemoryManager.allocate( this, nextExpandSize );
            m_SpaceLeft = nextExpandSize;
            trace( "[Query Memory Manager] - New head = ", m_CurrentPtr, ". Space Lef = ", m_SpaceLeft, " Bytes." );
         }
         void * tmp = m_CurrentPtr;
         if( m_CurrentPtr != nullptr ) {
            m_CurrentPtr = static_cast< char * >( m_CurrentPtr ) + p_AllocSize;
            m_SpaceLeft -= p_AllocSize;
         } else {
            m_GeneralMemoryManager.handle_error( );
            wtf( "[Query Memory Manager] - Could not aquire ", nextExpandSize, " Bytes ephimeral memory." );
         }
         trace( "[Query Memory Manager] - head = ", m_CurrentPtr, ". Space Lef = ", m_SpaceLeft, " Bytes." );
         return tmp;
      }

      void * allocate( MSV_PPUNUSED abstract_memory_manager * const p_Caller, MSV_PPUNUSED size_t p_AllocSize ) override {
         warn( "[Query Memory Manager] - Allocate with different Memory Manager should not be invoked on the Query Memory Manager." );
         return nullptr;
      }

      void deallocate( MSV_PPUNUSED abstract_memory_manager * const p_Caller, MSV_PPUNUSED void * const p_Ptr ) override {
         trace( "[Query Memory Manager] - IN.  ( Caller = ", p_Caller, ". Pointer = ", p_Ptr, " )." );
         info( "[Query Memory Manager] - Deallocate ( abstract_memory_manager * const, void * const ) ",
               "should not be invoked on the Query Memory Manager." );
      }

      void deallocate( MSV_PPUNUSED void * const p_Ptr ) override {
         trace( "[Query Memory Manager] - IN.  ( Pointer = ", p_Ptr, " )." );
         info( "[Query Memory Manager] - Deallocate ( void * const ) should not be invoked on the Query Memory Manager." );
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
