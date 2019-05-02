#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_PAGED_MM_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_PAGED_MM_H

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
        if (size <= PAGE_SIZE - static_cast<uint32_t>(header.m_currOffset)) {
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
        assert(size < PAGE_SIZE - sizeof(PageHeader));
        auto& manager = mmap_memory_manager::getInstance();

        void* loc = nullptr;
        if (current_page != nullptr)
            loc = current_page->allocate(size);
	
        if (loc == nullptr) {
            // TODO: handle next page allocation from chunk
            if (current_chunk == nullptr)
                current_chunk = manager.allocateContinuous();
            //ChunkHeader* header = reinterpret_cast<uint64_t>(current_chunk) - sizeof(ChunkHeader);
            loc = manager.allocate(PAGE_SIZE, current_chunk);
            if (loc == nullptr) {
                //TODO: handle concurrency
                current_chunk = manager.allocateContinuous();
                loc = manager.allocate(PAGE_SIZE, current_chunk);
                current_page = reinterpret_cast<Page*>(loc);
                return current_page->allocate(size);
            }
            else {
                return loc;
            }
        }
        
        return nullptr;
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
