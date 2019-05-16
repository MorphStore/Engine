#include <core/memory/management/abstract_mm.h>
#include <core/memory/management/mmap_mm.h>
#include <core/memory/management/paged_mm.h>

#include <cstdio>
#include <vector>
#include <chrono>
#include <iostream>

namespace morphstore {

mmap_memory_manager* mmap_memory_manager::m_Instance = nullptr;
paged_memory_manager* paged_memory_manager::global_manager = nullptr;

}

int main(int /*argc*/, char** /*argv*/) {
    morphstore::mmap_memory_manager& instance = morphstore::mmap_memory_manager::getInstance();

    void* ptr = reinterpret_cast<void*>(instance.allocateContinuous());
    //std::cout << "got address: " << ptr << std::endl;

    morphstore::ChunkHeader* header = reinterpret_cast<morphstore::ChunkHeader*>(reinterpret_cast<uint64_t>(ptr) - sizeof(morphstore::ChunkHeader));
    uint64_t state_test = header->getAllocBits64(60, 4);

    //std::cout << "State test is " << std::hex << state_test << std::endl;
    assert(state_test == 0b1011l);

    state_test = header->getAllocBits64(0, 1);
    //std::cout << "State test is " << std::hex << state_test << std::endl;
    assert(state_test == (1ul << 63));
    state_test = header->getAllocBits64(0, 4);
    //std::cout << "State test is " << std::hex << state_test << std::endl;

    morphstore::paged_memory_manager& page_instance = morphstore::paged_memory_manager::getGlobalInstance();
    /*page_instance.setCurrentChunk(ptr);

    void* obj_ptr = page_instance.allocate(32);
    page_instance.deallocate(obj_ptr);*/

    header = reinterpret_cast<morphstore::ChunkHeader*>( reinterpret_cast<uint64_t>(ptr) - sizeof(morphstore::ChunkHeader) );
    
    volatile uint64_t* bitmap_val = reinterpret_cast<volatile uint64_t*>(header->bitmap);
    volatile uint64_t* bitmap_end = reinterpret_cast<volatile uint64_t*>(header->getBitmapEnd());

    while (bitmap_val != bitmap_end) {
        assert( *bitmap_val == 0l);
        bitmap_val = reinterpret_cast<uint64_t*>( reinterpret_cast<uint64_t>(bitmap_val) + sizeof(uint64_t));
    }
    // performance tests
    
    const uint32_t ARRAY_LENGTH = 900000;
    uint32_t const object_sizes[7] = {8, 16, 32, 64, 128, 256, 512};
    //const uint32_t OBJ_SIZE = 32;
    void* ptrs[ARRAY_LENGTH];

    for (int j = 0; j<7; ++j) {
        std::cout << "Using object size " << object_sizes[j] << " bytes and " << ARRAY_LENGTH << " alloc and dealloc runs" << std::endl;
        ptr = reinterpret_cast<void*>(instance.allocateContinuous());
        page_instance.setCurrentChunk(ptr);

        auto start = std::chrono::system_clock::now();
        for (int k = 0; k < 3; ++k) {
            for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
                ptrs[i] = page_instance.allocate(object_sizes[j]);
                //std::cout << "Allocating on chunk " << std::hex << ( ~(morphstore::ALLOCATION_SIZE - 1) & reinterpret_cast<uint64_t>(ptrs[i]) ) << std::endl;
                //std::cout << "Iteration " << std::dec << i << std::endl;
                if (ptrs[i] == nullptr)
                    std::cerr << "Pointer " << i << " got returned as zero" << std::endl;
            }
            //std::cout << "Beginning deallocation" << std::endl;
            for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
                //std::cout << "it: " << i << std::endl;
                //std::cout << "Deallocating on chunk " << std::hex << ( ~(morphstore::ALLOCATION_SIZE - 1) & reinterpret_cast<uint64_t>(ptrs[i]) ) << std::endl;
                //std::cout << "deallocating " << ptrs[i] << std::endl;
                page_instance.deallocate(ptrs[i]);
            }
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        std::cout << "Needed " << elapsed_seconds.count() << " for " << ARRAY_LENGTH << " allocations and deallocations using the custom mm" << std::endl;

        start = std::chrono::system_clock::now();
        for (int k = 0; k < 3; ++k) {
            for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
                ptrs[i] = new char[object_sizes[j]];
                //std::cout << "Iteration " << std::dec << i << std::endl;
                if (ptrs[i] == nullptr)
                    std::cerr << "Pointer " << i << " got returned as zero" << std::endl;
            }
            //std::cout << "Beginning deallocation" << std::endl;
            for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
                //std::cout << "it: " << i << std::endl;
                delete[] reinterpret_cast<char*>(ptrs[i]);
            }
        }
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << "Needed " << elapsed_seconds.count() << " for " << ARRAY_LENGTH << " allocations and deallocations using std allocator" << std::endl;
    }
    assert(false);

    return 0;
}
