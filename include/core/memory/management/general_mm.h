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
//         trace( "[General Memory Manager] - IN.  ( void )." );
         static general_memory_manager instance;
         return instance;
      }

      general_memory_manager(general_memory_manager const &) = delete;

      general_memory_manager &operator=(general_memory_manager const &) = delete;

      ~general_memory_manager(void) {
         auto *handle = m_EphimeralMemoryBinHandler.get_tail();
         while (handle != nullptr) {
            stdlib_free(handle->m_BasePtr);
            handle = handle->prev();
         }
         handle = m_PerpetualMemoryBinHandler.get_tail();
         while (handle != nullptr) {
            stdlib_free(handle->m_BasePtr);
            handle = handle->prev();
         }
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
         if (!m_Initialized)
            handle_error();
      }

   public:

      void *allocate(size_t p_AllocSize) override {
         void *tmp = stdlib_malloc(p_AllocSize);
         if (tmp != nullptr) {
            m_PerpetualMemoryBinHandler.append_bin(this, tmp, p_AllocSize);
            return tmp;
         } else {
            handle_error();
            return nullptr;
         }
      }

      void *allocate(abstract_memory_manager *const p_Caller, size_t p_AllocSize) override {
         if (instanceof<general_memory_manager>(p_Caller)) {
            handle_error();
         }
         void *tmp = stdlib_malloc(p_AllocSize);
         if (tmp != nullptr) {
            m_EphimeralMemoryBinHandler.append_bin(p_Caller, tmp, p_AllocSize);
            return tmp;
         } else {
            handle_error();
            return nullptr;
         }
      }

      void deallocate(MSV_PPUNUSED abstract_memory_manager *const, MSV_PPUNUSED void *const) override {
         // NOP
      }

      void deallocate(MSV_PPUNUSED void *const) override {
         //@todo THIS CAN BE DONE!!! FOR PERPETUAL STORAGE
      }

      void handle_error(void) override {
         //@todo IMPLEMENT
         exit(1);
      }

      void destroy(abstract_memory_manager *const p_Caller) {
         if (instanceof<general_memory_manager>(p_Caller)) {
            handle_error();
         }
         auto handle = m_EphimeralMemoryBinHandler.find_last(p_Caller);
         if (handle == m_EphimeralMemoryBinHandler.get_tail()) {
            while (handle != nullptr) {
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
               handle =
                  m_EphimeralMemoryBinHandler.find_prev(
                     p_Caller,
                     m_EphimeralMemoryBinHandler.find_and_remove_reverse_until_first_other(p_Caller, handle)
                  );
            }
         }
      }

   private:
      bool m_Initialized;
      memory_bin_handler m_PerpetualMemoryBinHandler;
      memory_bin_handler m_EphimeralMemoryBinHandler;

   };

}}

#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_GENERAL_MM_H

