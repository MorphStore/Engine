#include <core/memory/management/abstract_mm.h>
#include <core/memory/management/mmap_mm.h>

#include <cstdio>
#include <vector>
#include <iostream>

namespace morphstore {

mmap_memory_manager* mmap_memory_manager::m_Instance = nullptr;

}

int main(int /*argc*/, char** /*argv*/) {
    morphstore::mmap_memory_manager& instance = morphstore::mmap_memory_manager::getInstance();

   char* ptr = reinterpret_cast<char*>(instance.allocateContinuous());

   (*ptr) = *ptr;

   return 0;
}
