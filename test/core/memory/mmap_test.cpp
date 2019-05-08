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

    morphstore::paged_memory_manager& page_instance = morphstore::paged_memory_manager::getGlobalInstance();
    page_instance.setCurrentChunk(ptr);
    
    const uint32_t ARRAY_LENGTH = 10000;
    void* ptrs[ARRAY_LENGTH];

    for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
        ptrs[i] = page_instance.allocate(4096);
        if (ptrs[i] == nullptr)
            std::cerr << "Pointer " << i << " got returned as zero" << std::endl;
    }

    std::cout << "Beginning deallocation" << std::endl;

    for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
        std::cout << "it: " << i << std::endl;
        page_instance.deallocate(ptrs[i]);
    }

    return 0;
}