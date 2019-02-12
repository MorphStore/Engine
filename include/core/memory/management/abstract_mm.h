/**
 * @file abstract_mm.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_ABSTRACT_MM_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_ABSTRACT_MM_H
namespace morphstore {

class abstract_memory_manager {
   public:
      virtual void *allocate(size_t) = 0;

      virtual void *allocate(abstract_memory_manager *const, size_t) = 0;

      virtual void deallocate(abstract_memory_manager *const, void *const) = 0;

      virtual void deallocate(void *const) = 0;

      virtual void handle_error() = 0;
};

}
#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_ABSTRACT_MM_H
