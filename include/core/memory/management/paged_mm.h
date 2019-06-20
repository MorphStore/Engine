#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_PAGED_MM_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_PAGED_MM_H

/*#include <core/memory/global/mm_hooks.h>
#include <core/memory/management/allocators/global_scope_allocator.h>
#include <core/utils/logger.h>*/

#include "core/memory/management/abstract_mm.h"
#include "core/memory/management/mmap_mm.h"

#include <string.h>

namespace morphstore {

static const size_t PAGE_SIZE = 1 << 15;

class paged_memory_manager;

class PageHeader {
    friend class Page;

public:
    PageHeader(uint32_t curr_offset)
        : m_currOffset(curr_offset), m_sumOffset(0), m_sema(), m_canDeallocate(false)
    {
        ////trace("called constructor");
    }

    void init()
    {
        m_currOffset = static_cast<uint32_t>(sizeof(PageHeader));
    }

    uint32_t m_currOffset;
    uint32_t m_sumOffset;
    std::mutex m_sema;
    bool m_canDeallocate;
};
static_assert( sizeof(PageHeader) > 0, "PageHeader must be larger than 0");

class Page {
public:
    Page() : header(sizeof(PageHeader) /*manager*/) {
        ////trace("called page constructor"); 
    }

    void* allocate(size_t size)
    {
        if (size <= PAGE_SIZE - static_cast<size_t>(header.m_currOffset)) {
            void* loc = reinterpret_cast<void*>(reinterpret_cast<uint64_t>(this) + static_cast<uint64_t>(header.m_currOffset));
            header.m_sumOffset += header.m_currOffset;
            header.m_currOffset += size;
            ////trace( "[PAGE] sum offset is now ", std::hex, header.m_sumOffset, ", current offset ", std::hex, header.m_currOffset);
            return loc;
        }
        else {
            header.m_canDeallocate = true;
            return nullptr;
        }
    }

    void deallocate(void* addr)
    {
        uint64_t offset = reinterpret_cast<uint64_t>(addr) - reinterpret_cast<uint64_t>(this);
        header.m_sumOffset -= offset;
        //trace( "[PAGE] sum offset is now ", std::hex, header.m_sumOffset, ", offset calculated ", std::hex, offset);

        if (header.m_sumOffset == 0 && header.m_canDeallocate) {
            //trace( "Triggered deallocation on ", this, " due to address ", addr);
            mmap_memory_manager::getInstance().deallocate(this);
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
        static paged_memory_manager instance;
        return instance;
    }

    paged_memory_manager() : current_page(nullptr), current_chunk(nullptr) {}

    void init()
    {
        current_chunk = mmap_memory_manager::getInstance().allocateContinuous();
    }

    void* allocate(size_t size) override
    {
        //TODO: throw exception
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
                ////trace("[PAGED_MM] Allocated new chunk on ", current_chunk);
            }
        
            //ChunkHeader* header = reinterpret_cast<uint64_t>(current_chunk) - sizeof(ChunkHeader);
            page_loc = manager.allocate(PAGE_SIZE, current_chunk);
            ////trace("[PAGED_MM] Allocated new page on ", page_loc);

            if (page_loc == nullptr) {
                //TODO: handle concurrency
                //FIXME: not necessary
                current_chunk = manager.allocateContinuous();
                page_loc = manager.allocate(PAGE_SIZE, current_chunk);
                current_page = reinterpret_cast<Page*>(page_loc);
                //trace("should construct");
                new (page_loc) Page();
                object_loc = current_page->allocate(size);
                if (object_loc == nullptr)
                    throw std::runtime_error("Allocation failed");

                return object_loc;
            }
            else {
                Page* page = new (page_loc) Page();
                current_page = page;
                object_loc = page->allocate(size);
                ////trace("Allocated object on ", object_loc, " with size ", std::hex, size);
                if (object_loc == nullptr)
                    throw std::runtime_error("Allocation failed");

                return object_loc;
            }
        }
        else {
            ////trace( "[PAGED_MM] Found object allocation spot in current page ", current_page);
            return object_loc;
        }
    }

    inline Page* getPage(void* addr)
    {
        return reinterpret_cast<Page*>( reinterpret_cast<uint64_t>(addr) & ~(DB_PAGE_SIZE - 1) );
    }

    void deallocate(void* addr) override
    {
        Page* page = getPage(addr);
        page->deallocate(addr); 
    } 
   
    //TODO: set protected and only available to test 
    void setCurrentChunk(void* chunk)
    {
        current_page = nullptr;
        current_chunk = chunk;
    }
    
    void *allocate(abstract_memory_manager *const /*manager*/, size_t /*size*/) override
    {
         throw std::runtime_error("Not implemented");
         return nullptr;
    }

    void deallocate(abstract_memory_manager *const /*manager*/, void *const /*ptr*/) override
    {
        throw std::runtime_error("Not implemented");
    }

    void * reallocate(abstract_memory_manager * const /*manager*/, void * /*ptr*/, size_t /*size*/) override
    {
        throw std::runtime_error("Not implemented");
        return nullptr;
    }

    void * reallocate(void* ptr, size_t size) override
    {
        //TODO: do proper impl
        void* ret = allocate(size);
        if (ret == nullptr) {
            throw std::runtime_error("Reallocation failed");
            return nullptr;
        }
        ObjectInfo* info = reinterpret_cast<ObjectInfo*>(ptr);
        memcpy(ret, ptr, info->size > size ? size : info->size); 
        deallocate(ptr);
        info = reinterpret_cast<ObjectInfo*>(ret);
        info->size = size;
        return ret;
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
