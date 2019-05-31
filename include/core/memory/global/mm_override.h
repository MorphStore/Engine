#ifndef MORPHSTORE_CORE_MEMORY_GLOBAL_MM_OVERRIDE_H
#define MORPHSTORE_CORE_MEMORY_GLOBAL_MM_OVERRIDE_H

#include <core/memory/management/mmap_mm.h>
#include <core/memory/management/paged_mm.h>

inline void* mm_malloc(size_t p_AllocSize) {
    size_t abs_needed_size = p_AllocSize + sizeof(morphstore::ObjectInfo);

    if (abs_needed_size > morphstore::ALLOCATION_SIZE) { // already allocates object info for convenience, aligned to chunksize 
        return morphstore::mmap_memory_manager::getInstance().allocateLarge(p_AllocSize);
    }
    else if (abs_needed_size > (( morphstore::DB_PAGE_SIZE ) - sizeof(morphstore::PageHeader) )) {
        morphstore::ObjectInfo* info = reinterpret_cast<morphstore::ObjectInfo*>( morphstore::mmap_memory_manager::getInstance().allocate(abs_needed_size) );
        info->size = abs_needed_size;
        return reinterpret_cast<void*>( reinterpret_cast<uint64_t>(info) + sizeof(morphstore::ObjectInfo));
    }
    else {
        morphstore::ObjectInfo* info = reinterpret_cast<morphstore::ObjectInfo*>(morphstore::paged_memory_manager::getGlobalInstance().allocate(abs_needed_size));
        info->size = abs_needed_size;
        return reinterpret_cast<void*>( reinterpret_cast<uint64_t>(info) + sizeof(morphstore::ObjectInfo));
    }

}

inline void* mm_realloc(void* p_Ptr, size_t p_AllocSize) {
    morphstore::ObjectInfo* info = reinterpret_cast<morphstore::ObjectInfo*>( reinterpret_cast<uint64_t>(p_Ptr) - sizeof(morphstore::ObjectInfo));

    if (info->size > morphstore::ALLOCATION_SIZE) {
        return morphstore::mmap_memory_manager::getInstance().reallocate(p_Ptr, p_AllocSize);
    }
    else if (info->size > (( morphstore::DB_PAGE_SIZE - sizeof(morphstore::PageHeader) ))) {
        return morphstore::mmap_memory_manager::getInstance().reallocate(info, p_AllocSize + sizeof(morphstore::ObjectInfo));
    }
    else {
        return morphstore::paged_memory_manager::getGlobalInstance().reallocate(info, p_AllocSize + sizeof(morphstore::ObjectInfo));
    }
}

inline void mm_free(void* p_FreePtr) {
    morphstore::ObjectInfo* info = reinterpret_cast<morphstore::ObjectInfo*>( reinterpret_cast<uint64_t>(p_FreePtr) - sizeof(morphstore::ObjectInfo));

    if (info->size > morphstore::ALLOCATION_SIZE) {
        morphstore::mmap_memory_manager::getInstance().deallocateLarge(p_FreePtr);
    }
    else if (info->size > (( morphstore::DB_PAGE_SIZE - sizeof(morphstore::PageHeader) ))) {
        morphstore::mmap_memory_manager::getInstance().deallocate(info);
    }
    else {
        morphstore::paged_memory_manager::getGlobalInstance().deallocate(info);
    }
}

#endif