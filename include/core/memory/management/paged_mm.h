#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_PAGED_MM_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_PAGED_MM_H

/*#include <core/memory/global/mm_hooks.h>
#include <core/memory/management/allocators/global_scope_allocator.h>
#include <core/utils/logger.h>*/

#include "core/memory/management/abstract_mm.h"
#include "core/memory/management/mmap_mm.h"

namespace morphstore {

static const size_t PAGE_SIZE = 1 << 14;

class paged_memory_manager;

class PageHeader {
    friend class Page;

public:
    PageHeader(uint32_t curr_offset, abstract_memory_manager& manager)
        : m_currOffset(curr_offset), m_sumOffset(0), m_sema(), m_manager(manager) {}

    uint32_t m_currOffset;
    uint32_t m_sumOffset;
    std::mutex m_sema;
    abstract_memory_manager& m_manager;
};

class Page {
public:
    Page(abstract_memory_manager& manager) : header(sizeof(PageHeader), manager) {}

    void* allocate(size_t size)
    {
        trace("Trying to allocate ", std::hex, size, " bytes in Page");
        trace("Current offset ", std::hex, header.m_currOffset, " out of ", std::hex, PAGE_SIZE);
        if (size <= PAGE_SIZE - static_cast<size_t>(header.m_currOffset)) {
            void* loc = reinterpret_cast<void*>(reinterpret_cast<uint64_t>(this) + static_cast<uint64_t>(header.m_currOffset));
            header.m_currOffset += size;
            header.m_sumOffset += header.m_currOffset;
            return loc;
        }
        else {
            return nullptr;
        }
    }

    void deallocate(void* addr)
    {
        uint64_t offset = reinterpret_cast<uint64_t>(addr) - reinterpret_cast<uint64_t>(this);
        header.m_sumOffset -= offset;

        if (header.m_sumOffset == 0) {
            header.m_manager.deallocate(this);
        }
    }

private:
    PageHeader header;
};

// No reuse slot page allocator
class paged_memory_manager : public abstract_memory_manager
{
public:
    static paged_memory_manager* global_manager;
    static paged_memory_manager& getGlobalInstance()
    {
        if (global_manager == nullptr) {
            global_manager = ::new paged_memory_manager();
        }
        return *global_manager;
    }

    paged_memory_manager() : current_page(nullptr), current_chunk(nullptr) {}

    void init()
    {
        current_chunk = mmap_memory_manager::getInstance().allocateContinuous();
    }

    void* allocate(size_t size) override
    {
        //TODO: throw exception
        trace("Allocation called");
        assert(size < PAGE_SIZE - sizeof(PageHeader) - sizeof(ObjectInfo));
        auto& manager = mmap_memory_manager::getInstance();
        size += sizeof(ObjectInfo); // Additional space for type and allocation information

        void* object_loc = nullptr;
        void* page_loc = nullptr;

        if (current_page != nullptr) {
            object_loc = current_page->allocate(size);
            //trace("[PAGED_MM] Allocated memory on spot ", object_loc, " from page ", current_page);
        }

        //current page was (probably) full    
        if (object_loc == nullptr) {
            // TODO: handle next page allocation from chunk
            if (current_chunk == nullptr) {
                current_chunk = manager.allocateContinuous();
                //trace("[PAGED_MM] Allocated new chunk on ", current_chunk);
            }
        
            //ChunkHeader* header = reinterpret_cast<uint64_t>(current_chunk) - sizeof(ChunkHeader);
            page_loc = manager.allocate(PAGE_SIZE, current_chunk);
            //trace("[PAGED_MM] Allocated new page on ", page_loc);

            if (page_loc == nullptr) {
                //TODO: handle concurrency
                //FIXME: not necessary
                current_chunk = manager.allocateContinuous();
                page_loc = manager.allocate(PAGE_SIZE, current_chunk);
                current_page = reinterpret_cast<Page*>(page_loc);
                object_loc = current_page->allocate(size);
                //1trace("[PAGED_MM] Allocated object on ", object_loc);
                return object_loc;
            }
            else {
                Page* page = reinterpret_cast<Page*>(page_loc);
                current_page = page;
                object_loc = page->allocate(size);
                //trace("Allocated object on ", object_loc, " with size ", std::hex, size);
                return object_loc;
            }
        }
        else {
            trace( "[PAGED_MM] Found object allocation spot in current page ", current_page);
            return object_loc;
        }
    }
    
    void deallocate(void* addr) override
    {
        mmap_memory_manager::getInstance().deallocate(addr);
    } 
   
    //TODO: set protected and only available to test 
    void setCurrentChunk(void* chunk)
    {
        current_chunk = chunk;
    }
    
    void *allocate(abstract_memory_manager *const /*manager*/, size_t /*size*/) override
    {
         return nullptr;
    }

    void deallocate(abstract_memory_manager *const /*manager*/, void *const /*ptr*/) override
    {

    }

    void * reallocate(abstract_memory_manager * const /*manager*/, void * /*ptr*/, size_t /*size*/) override
    {
        return nullptr;
    }

    void * reallocate(void* /*ptr*/, size_t /*size*/) override
    {
        return nullptr;
    }

    void handle_error() override
    {

    }

private:
    Page* current_page;
    void* current_chunk; 
    std::mutex sema;
};


} //namespace morphstore

#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_PAGED_MM_H
