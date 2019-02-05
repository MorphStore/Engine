/**
 * @file general_mm.h
 * @brief Brief description
 * @author Johannes Pietrzyk
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

#include "../../utils/basic_types.h"
#include "../../utils/helper.h"
#include "../../utils/preprocessor.h"

#include "utils/memory_bin_handler.h"

namespace morphstore { namespace memory {

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

      ~general_memory_manager(void) {
         trace( "[General Memory Manager] - IN.  ( void )." );
         auto * handle = m_EphimeralMemoryBinHandler.get_tail( );
         auto * rootEphimeral = m_EphimeralMemoryBinHandler.get_root( );
         while( handle != rootEphimeral ) {
            debug( "[General Memory Manager] - Freeing ephimeral memory ", handle->m_BasePtr, " in Bin ", handle, "." );
            stdlib_free(handle->m_BasePtr);
            handle = handle->prev();
         }
         handle = m_PerpetualMemoryBinHandler.get_tail();
         auto * rootPerpetual = m_PerpetualMemoryBinHandler.get_root( );
         while (handle != rootPerpetual) {
            debug( "[General Memory Manager] - Freeing perpetual memory ", handle->m_BasePtr, " in Bin ", handle, "." );
            stdlib_free(handle->m_BasePtr);
            handle = handle->prev();
         }
         trace( "[General Memory Manager] - OUT. ( void )." );
      }

   private:
      general_memory_manager(void) :
         m_Initialized{(
                          (morphstore::memory::stdlib_malloc_ptr == nullptr) ||
                          (morphstore::memory::stdlib_malloc_ptr == nullptr) ||
                          (morphstore::memory::stdlib_malloc_ptr == nullptr)
                       ) ? init_mem_hooks() : true
         },
         m_PerpetualMemoryBinHandler(this),
         m_EphimeralMemoryBinHandler(this) {
         trace( "[General Memory Manager] - IN.  ( void )." );
         if (!m_Initialized)
            handle_error();
         trace(
            "[General Memory Manager] - OUT. ( this: ", this,
            ". Perpetual Memory Bin Handler: ", &m_PerpetualMemoryBinHandler ,
            ". Ephimeral Memory Bin Handler: ", &m_EphimeralMemoryBinHandler , " ).");
      }

   public:

      void *allocate(size_t p_AllocSize) override {
         trace( "[General Memory Manager] - IN.  ( AllocSize = ", p_AllocSize, " )." );
         void * tmp = stdlib_malloc( p_AllocSize + MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE );

         debug( "[General Memory Manager] - Allocated ", p_AllocSize + MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE, " Bytes ( @ position: ", tmp, " ).");
         if (tmp != nullptr) {
            debug( "[General Memory Manager] - Add space to perpetual Memory Bin Handler." );
            m_PerpetualMemoryBinHandler.append_bin(this, tmp, p_AllocSize);

            size_t tmpToSizeT = reinterpret_cast< size_t >( tmp );
            size_t const offset = tmpToSizeT & MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE;
            if (offset)
               tmp = reinterpret_cast< void * >(
                  tmpToSizeT + ( MSV_MEMORY_MANAGER_ALIGNMENT_BYTE - offset ) );

            trace( "[General Memory Manager] - OUT. ( Aligned Pointer: ", tmp, " )." );
            return tmp;
         } else {
            wtf( "[General Memory Manager] - allocate( AllocSize = ", p_AllocSize, " ). Could not allocate the memory." );
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
         void * tmp = stdlib_malloc( p_AllocSize + MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE );
         debug( "[General Memory Manager] - Allocated ", p_AllocSize + MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE, " Bytes ( @ position: ", tmp, " ).");
         if (tmp != nullptr) {
            debug( "[General Memory Manager] - Add space to ephimeral Memory Bin Handler." );
            m_EphimeralMemoryBinHandler.append_bin(p_Caller, tmp, p_AllocSize);

            size_t tmpToSizeT = reinterpret_cast< size_t >( tmp );
            size_t const offset = tmpToSizeT & MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE;
            if (offset)
               tmp = reinterpret_cast< void * >(
                  tmpToSizeT + ( MSV_MEMORY_MANAGER_ALIGNMENT_BYTE - offset ) );

            trace( "[General Memory Manager] - OUT. ( Aligned Pointer: ", tmp, " )." );
            return tmp;
         } else {
            wtf( "[General Memory Manager] - allocate( Caller = ", p_Caller, ". AllocSize = ", p_AllocSize, " ): Could not allocate ", p_AllocSize, " Bytes." );
            handle_error( );
            trace( "[General Memory Manager] - OUT. ( nullptr )." );
            return nullptr;
         }
      }

      void deallocate(MSV_PPUNUSED abstract_memory_manager *const p_Caller, MSV_PPUNUSED void *const p_Ptr ) override {
         trace( "[General Memory Manager] - IN.  ( Caller = ", p_Caller, ". Pointer = ", p_Ptr, " )." );
         info( "[General Memory Manager] - Deallocate should not be invoked on the General Memory Manager." );
         // NOP
      }

      void deallocate(MSV_PPUNUSED void *const p_Ptr ) override {
         trace( "[General Memory Manager] - IN.  ( Pointer = ", p_Ptr, " )." );
         warn( "[General Memory Manager] - @TODO: This can be done for perpetual storage. Needed to be implemented." );
         //@todo THIS CAN BE DONE!!! FOR PERPETUAL STORAGE
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
         auto handle = m_EphimeralMemoryBinHandler.find_last(p_Caller);
         if (handle == m_EphimeralMemoryBinHandler.get_tail()) {
            while (handle != nullptr) {
               trace("[General Memory Manager] - Freeing Memory ( ", handle->m_BasePtr, " ) from Bin ( ", handle, " )." );
               auto tmpHandle =
                  m_EphimeralMemoryBinHandler.find_and_remove_reverse_until_first_other(p_Caller, handle);
               m_EphimeralMemoryBinHandler.set_tail(tmpHandle);
               handle =
                  m_EphimeralMemoryBinHandler.find_prev(
                     p_Caller,
                     tmpHandle);
            }
         } else {
            while (handle != nullptr) {
               trace("[General Memory Manager] - Freeing Bin from Caller ( ", handle->m_BasePtr, " ). ");
               handle =
                  m_EphimeralMemoryBinHandler.find_prev(
                     p_Caller,
                     m_EphimeralMemoryBinHandler.find_and_remove_reverse_until_first_other(p_Caller, handle)
                  );
            }
         }
         trace( "[General Memory Manager] - OUT. ( void )." );
      }

   private:
      bool m_Initialized;
      memory_bin_handler m_PerpetualMemoryBinHandler;
      memory_bin_handler m_EphimeralMemoryBinHandler;

   };

}}

#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_GENERAL_MM_H

