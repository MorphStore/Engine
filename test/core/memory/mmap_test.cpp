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
    void* page_ptr = page_instance.allocate(4000);
    std::cout << "got address for small object: " << page_ptr << std::endl;
        
    return 0;
}
