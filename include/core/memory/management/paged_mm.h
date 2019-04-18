#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_PAGED_MM_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_PAGED_MM_H

#include "core/memory/management/mmap_mm.h"

namespace morphstore {

class PageHeader {
    friend class Page;

public:
    PageHeader(uint32_t curr_offset, paged_memory_manager& manager)
        : m_currOffset(curr_offset), m_sumOffset(0), m_sema(), m_manager(manager) {}

    uint32_t m_currOffset;
    uint32_t m_sumOffset;
    std::mutex m_sema;
    paged_memory_manager& m_manager;
};

class Page {
public:
    Page(paged_memory_manager& manager) : header(sizeof(PageHeader), manager) {}

    void* allocate(size_t size)
    {
        if (size <= paged_memory_manager::PAGE_SIZE - static_cast<uint32_t>(header->curr_offset)) {
            void* loc = reinterpret_cast<char*>(this) + reinterpret_cast<char*>(header->curr_offset);
            curr_offset += size;
            sum_offset += curr_offset;
            return loc;
        }
        else {
            return nullptr;
        }
    }

    void deallocate(void* addr)
    {
        uint32_t offset = reinterpret_cast<char*>(addr) - reinterpret_cast<uint64_t>(this);
        sum_offset -= offset;

        if (sum_offset == 0) {
            header->m_manager.deallocate(this);
        }
    }

private:
    PageHeader header;
};

// No reuse slot page allocator
class paged_memory_manager : public abstract_memory_manager
{
public:
    static const size_t PAGE_SIZE = 1 << 14;

    paged_memory_manager() : current_page(nullptr), current_chunk(nullptr) {}

    void init()
    {
        current_chunk = mmap_memory_manager::getInstance().allocateContinuous();
    }

    void* allocate(size_t size)
    {
        //TODO: throw exception
        assert(size < PAGE_SIZE - sizeof(PageHeader));

        void* loc current_page->allocate(size);
        if (loc == nullptr) {
            // TODO: handle next page allocation from chunk

        }
    }
    
    void deallocate(void* addr)
    {
        //TODO:
    } 

private:
    Page* current_page;
    void* current_chunk; 
    std::mutex sema;
};

} //namespace morphstore

#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_PAGED_MM_H
