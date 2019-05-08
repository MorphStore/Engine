#include <core/memory/management/abstract_mm.h>
#include <core/memory/management/mmap_mm.h>
#include <core/memory/management/paged_mm.h>

#include <cstdio>
#include <vector>
#include <iostream>

namespace morphstore {

mmap_memory_manager* mmap_memory_manager::m_Instance = nullptr;
paged_memory_manager* paged_memory_manager::global_manager = nullptr;

}

int main(int /*argc*/, char** /*argv*/) {
    morphstore::mmap_memory_manager& instance = morphstore::mmap_memory_manager::getInstance();

    void* ptr = reinterpret_cast<void*>(instance.allocateContinuous());
    std::cout << "got address: " << ptr << std::endl;

    morphstore::ChunkHeader* header = reinterpret_cast<morphstore::ChunkHeader*>(reinterpret_cast<uint64_t>(ptr) - sizeof(morphstore::ChunkHeader));
    uint64_t state_test = header->getAllocBits64(60, 4);

    std::cout << "State test is " << std::hex << state_test << std::endl;
    assert(state_test == 0b1011l);

    state_test = header->getAllocBits64(0, 1);
    std::cout << "State test is " << std::hex << state_test << std::endl;
    assert(state_test == (1ul << 63));
    state_test = header->getAllocBits64(0, 4);
    std::cout << "State test is " << std::hex << state_test << std::endl;

    morphstore::paged_memory_manager& page_instance = morphstore::paged_memory_manager::getGlobalInstance();
    page_instance.setCurrentChunk(ptr);
    
    const uint32_t ARRAY_LENGTH = 225;
    void* ptrs[ARRAY_LENGTH];

    for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
        ptrs[i] = page_instance.allocate(4096);
        std::cout << "Iteration " << std::dec << i << std::endl;
        if (ptrs[i] == nullptr)
            std::cerr << "Pointer " << i << " got returned as zero" << std::endl;
    }

    std::cout << "Beginning deallocation" << std::endl;

    for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
        std::cout << "it: " << i << std::endl;
        page_instance.deallocate(ptrs[i]);
    }
    assert(false);

    return 0;
}
