/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/


#ifndef QUEUEBENCHMARK_INCLUDE_MORPHSTORE_INCLUDE_CORE_MEMORY_MEMORYMANAGER_H
#define QUEUEBENCHMARK_INCLUDE_MORPHSTORE_INCLUDE_CORE_MEMORY_MEMORYMANAGER_H

#include <stdlibs>
#include "DefaultAllocator.h"
#include "MemoryAllocator.h"

namespace morphstore {
    
    class MemoryManager {
        using pointer = void*;
        using allocator_t = DefaultAllocator<uint8_t>;
        BaseMemoryAllocator * memoryAllocator;
        
        static MemoryManager * instance;
      
      public:
        
        MemoryManager() : memoryAllocator(new MemoryAllocator()){
            ///
        }
        
        MemoryManager(BaseMemoryAllocator * memoryAllocator) : memoryAllocator(memoryAllocator) {
            ///
        }
        
        pointer allocate(std::size_t size){
            return memoryAllocator->allocate(size);
        }
        
        void deallocate(pointer p, std::size_t size){
            memoryAllocator->deallocate(p, size);
        }
        
        static MemoryManager * getInstance(){
            if(instance == nullptr)
                instance = new MemoryManager();
            return instance;
        }
        
        static void setMemoryAllocator(BaseMemoryAllocator * memoryAllocator){
            delete getInstance()->memoryAllocator;
            getInstance()->memoryAllocator = memoryAllocator;
        }
        
        static pointer staticAllocate(size_t size){
            return getInstance()->allocate(size);
        }
        
        static void staticDeallocate(pointer p, std::size_t size){
            getInstance()->deallocate(p, size);
        }
    };
    
}
#endif //QUEUEBENCHMARK_INCLUDE_MORPHSTORE_INCLUDE_CORE_MEMORY_MEMORYMANAGER_H
