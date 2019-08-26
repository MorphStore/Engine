#ifndef MORPHSTORE_CORE_MEMORY_GLOBAL_MM_OVERRIDE_H
#define MORPHSTORE_CORE_MEMORY_GLOBAL_MM_OVERRIDE_H

#include <core/memory/management/mmap_mm.h>
#include <core/memory/management/paged_mm.h>

namespace morphstore {

uint64_t __malloc_count = 0;

inline void* mm_malloc(size_t p_AllocSize) {
    ++__malloc_count;
    size_t abs_needed_size = p_AllocSize + sizeof(morphstore::std_mmap_mm::ObjectInfo);

    if (abs_needed_size > morphstore::std_mmap_mm::ALLOCATION_SIZE) { // already allocates object info for convenience, aligned to chunksize 
        return morphstore::std_mmap_mm::getInstance().allocateLarge(p_AllocSize);
    }
    else if (abs_needed_size > (( morphstore::std_mmap_mm::DB_PAGE_SIZE ) - sizeof(morphstore::PageHeader) )) {
        morphstore::std_mmap_mm::ObjectInfo* info = reinterpret_cast<morphstore::std_mmap_mm::ObjectInfo*>( morphstore::std_mmap_mm::getInstance().allocate(abs_needed_size) );
        info->size = abs_needed_size;
        return reinterpret_cast<void*>( reinterpret_cast<uint64_t>(info) + sizeof(morphstore::std_mmap_mm::ObjectInfo));
    }
    else {
        morphstore::std_mmap_mm::ObjectInfo* info = reinterpret_cast<morphstore::std_mmap_mm::ObjectInfo*>(morphstore::paged_memory_manager::getGlobalInstance().allocate(abs_needed_size));
        info->size = abs_needed_size;
        return reinterpret_cast<void*>( reinterpret_cast<uint64_t>(info) + sizeof(morphstore::std_mmap_mm::ObjectInfo));
    }

}

inline void* mm_realloc(void* p_Ptr, size_t p_AllocSize) {

    if (p_Ptr == nullptr)
        return mm_malloc(p_AllocSize);

    if (p_AllocSize == 0)
        return nullptr;

    morphstore::std_mmap_mm::ObjectInfo* info = reinterpret_cast<morphstore::std_mmap_mm::ObjectInfo*>( reinterpret_cast<uint64_t>(p_Ptr) - sizeof(morphstore::std_mmap_mm::ObjectInfo));

    if (info->size > morphstore::std_mmap_mm::ALLOCATION_SIZE) {
        return morphstore::std_mmap_mm::getInstance().reallocate(p_Ptr, p_AllocSize);
    }
    else if (info->size > (( morphstore::std_mmap_mm::DB_PAGE_SIZE - sizeof(morphstore::PageHeader) ))) {
        return morphstore::std_mmap_mm::getInstance().reallocate(info, p_AllocSize + sizeof(morphstore::std_mmap_mm::ObjectInfo));
    }
    else {
        void* ptr = morphstore::paged_memory_manager::getGlobalInstance().reallocate(info, p_AllocSize + sizeof(morphstore::std_mmap_mm::ObjectInfo));
        morphstore::std_mmap_mm::ObjectInfo* info = reinterpret_cast<morphstore::std_mmap_mm::ObjectInfo*>(ptr);
        assert(info->size == p_AllocSize + sizeof(morphstore::std_mmap_mm::ObjectInfo));
        return &(info[1]);
    }

}

inline void mm_free(void* p_FreePtr) {
    morphstore::std_mmap_mm::ObjectInfo* info = reinterpret_cast<morphstore::std_mmap_mm::ObjectInfo*>( reinterpret_cast<uint64_t>(p_FreePtr) - sizeof(morphstore::std_mmap_mm::ObjectInfo));

    if( MSV_CXX_ATTRIBUTE_LIKELY( morphstore::paged_memory_manager_state_helper::get_instance( ).is_alive( ) ) )
        if (info->size > morphstore::std_mmap_mm::ALLOCATION_SIZE) {
            morphstore::std_mmap_mm::getInstance().deallocateLarge(p_FreePtr);
        }
        else if (info->size > (( morphstore::std_mmap_mm::DB_PAGE_SIZE - sizeof(morphstore::PageHeader) ))) {
            morphstore::std_mmap_mm::getInstance().deallocate(info);
        }
        else {
            morphstore::paged_memory_manager::getGlobalInstance().deallocate(info);
        }
    else {
        //morphstore::stdlib_free_ptr( p_FreePtr );
    }
}

} // namespace morphstore
#endif
