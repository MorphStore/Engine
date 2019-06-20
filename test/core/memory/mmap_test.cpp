//#define USE_MMAP_MM

#include <core/memory/management/abstract_mm.h>
#include <core/memory/management/mmap_mm.h>
#include <core/memory/management/paged_mm.h>

//#include <core/memory/global/mm_hooks.h>
#include <core/memory/morphstore_mm.h>
/*#include <core/memory/management/allocators/global_scope_allocator.h>
#include <core/memory/management/query_mm.h>*/
//#include <core/memory/global/mm_override.h>
    // load capacity output

#include <cstdio>
#include <vector>
#include <chrono>
#include <iostream>
#include <map>


namespace morphstore {

mmap_memory_manager* mmap_memory_manager::m_Instance = nullptr;
paged_memory_manager* paged_memory_manager::global_manager = nullptr;

}

int main(int /*argc*/, char** /*argv*/) {
    init_mem_hooks();
    morphstore::mmap_memory_manager& instance = morphstore::mmap_memory_manager::getInstance();

    void* ptr = reinterpret_cast<void*>(instance.allocateContinuous());

    morphstore::ChunkHeader* header = reinterpret_cast<morphstore::ChunkHeader*>(reinterpret_cast<uint64_t>(ptr) - sizeof(morphstore::ChunkHeader));
    uint64_t state_test = header->getAllocBits64(60, 4);

    assert(state_test == 0b1011l);

    state_test = header->getAllocBits64(0, 1);
    assert(state_test == (1ul << 63));
    state_test = header->getAllocBits64(0, 4);

    morphstore::paged_memory_manager& page_instance = morphstore::paged_memory_manager::getGlobalInstance();

    void* obj_ptr = mm_malloc(32);
    *reinterpret_cast<uint64_t*>(obj_ptr) = 64;
    obj_ptr = mm_realloc(obj_ptr, 64);

    reinterpret_cast<uint64_t*>(obj_ptr)[5] = 64;
    std::cout << "Current val: " << *reinterpret_cast<uint64_t*>(obj_ptr) << std::endl;
    assert(*reinterpret_cast<uint64_t*>(obj_ptr) == 64);

    obj_ptr = mm_realloc(obj_ptr, 8);
    assert(*reinterpret_cast<uint64_t*>(obj_ptr) == 64);
    mm_free(obj_ptr);
    std::cout << "Done realloc" << std::endl;

    unsigned inDataCount = 234000;
    uint64_t* columns[128];
    for (int j = 0; j < 128; j+=2) {
        columns[j] = reinterpret_cast<uint64_t*>( mm_malloc(inDataCount * sizeof(uint64_t)) );
        columns[j+1] = reinterpret_cast<uint64_t*>( mm_malloc(inDataCount * sizeof(uint64_t)) );
        
        std::map<std::pair<uint64_t, uint64_t>, uint64_t> groupIds;
        for(unsigned i = 0; i < inDataCount; i++) {
            std::pair<uint64_t,uint64_t> pair = std::make_pair(columns[j][i], columns[j+1][i]);
            uint64_t & groupId = groupIds[pair];
            if(!groupId) { // The data item(key) was not found.
                groupId = groupIds.size();
            }
        }
    }

    for (int j = 0; j < 128; j++) {
        delete[] columns[j];
    }


    //obj_ptr = malloc(72704);
    //free(obj_ptr);*/

    header = reinterpret_cast<morphstore::ChunkHeader*>( reinterpret_cast<uint64_t>(ptr) - sizeof(morphstore::ChunkHeader) );
    
    volatile uint64_t* bitmap_val = reinterpret_cast<volatile uint64_t*>(header->bitmap);
    volatile uint64_t* bitmap_end = reinterpret_cast<volatile uint64_t*>(header->getBitmapEnd());

    while (bitmap_val != bitmap_end) {
        assert( *bitmap_val == 0l);
        bitmap_val = reinterpret_cast<uint64_t*>( reinterpret_cast<uint64_t>(bitmap_val) + sizeof(uint64_t));
    }
    // performance tests
    
    const uint32_t ARRAY_LENGTH = 1000000;
    uint32_t const object_sizes[7] = {8, 16, 32, 64, 128, 256, 512};
    //const uint32_t OBJ_SIZE = 32;
    void* ptrs[ARRAY_LENGTH];

    for (int j = 0; j<7; ++j) {
        std::cout << "Using object size " << object_sizes[j] << " bytes and " << ARRAY_LENGTH << " alloc and dealloc runs" << std::endl;
        ptr = reinterpret_cast<void*>(instance.allocateContinuous());
        page_instance.setCurrentChunk(ptr);

        std::chrono::time_point<std::chrono::system_clock> alloc_start, alloc_end, dealloc_start, dealloc_end;

        auto start = std::chrono::system_clock::now();
        for (int k = 0; k < 3; ++k) {

            alloc_start = std::chrono::system_clock::now();
            for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
                ptrs[i] = mm_malloc(object_sizes[j]);
                if (ptrs[i] == nullptr)
                    std::cerr << "Pointer " << i << " got returned as zero" << std::endl;
                *reinterpret_cast<uint64_t*>(ptrs[i]) = 4;
            }
            alloc_end = std::chrono::system_clock::now();

            dealloc_start = std::chrono::system_clock::now();
            for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
                mm_free(reinterpret_cast<char*>(ptrs[i]));
            }
            dealloc_end = std::chrono::system_clock::now();

        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        std::chrono::duration<double> elapsed_sec_alloc = alloc_end - alloc_start;
        std::chrono::duration<double> elapsed_sec_dealloc = dealloc_end - dealloc_start;
        std::cout << "Alloc: " << elapsed_sec_alloc.count() << ", dealloc: " << elapsed_sec_dealloc.count() << std::endl;
        std::cout << "Needed " << elapsed_seconds.count() << " for " << ARRAY_LENGTH << " allocations and deallocations using the custom mm" << std::endl;

        start = std::chrono::system_clock::now();
        for (int k = 0; k < 3; ++k) {

            alloc_start = std::chrono::system_clock::now();
            for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
                ptrs[i] = morphstore::stdlib_malloc(object_sizes[j]);
                if (ptrs[i] == nullptr)
                    std::cerr << "Pointer " << i << " got returned as zero" << std::endl;
                *reinterpret_cast<uint64_t*>(ptrs[i]) = 4;
            }
            alloc_end = std::chrono::system_clock::now();
            dealloc_start = std::chrono::system_clock::now();
            for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
                morphstore::stdlib_free(reinterpret_cast<char*>(ptrs[i]));
            }
            dealloc_end = std::chrono::system_clock::now();
        }
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        elapsed_sec_alloc = alloc_end - alloc_start;
        elapsed_sec_dealloc = dealloc_end - dealloc_start;
        std::cout << "Alloc: " << elapsed_sec_alloc.count() << ", dealloc: " << elapsed_sec_dealloc.count() << std::endl;
        std::cout << "Needed " << elapsed_seconds.count() << " for " << ARRAY_LENGTH << " allocations and deallocations using std allocator" << std::endl;
    }
    assert(false);

    return 0;
}
