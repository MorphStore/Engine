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


#ifndef MORPHSTORE_INCLUDE_CORE_MEMORY_COLUMNALLOCATOR_H
#define MORPHSTORE_INCLUDE_CORE_MEMORY_COLUMNALLOCATOR_H

#include <stdlibs>

namespace morphstore {
    
    /**
     * This Allocator extends the std::allocator by static allocate and deallocate functions
     * used by new and delete operators of morphstore classes.
     * Size of allocation depends on the data type T and the number of elements num.
     * I.e. a type with size of 1 byte (e.g. char) and num of 5 results in an allocation of 5 bytes,
     * but a type with size of 4 bytes (e.g. int32) and num of 5 results in 20 bytes allocated.
     * @tparam T data type for allocation.
     */
    template< typename T >
    class DefaultAllocator : public std::allocator<T> {
      
      public:
        using value_type = T;
        using pointer = value_type *;
        using const_pointer = const pointer;
        
        
        pointer allocate(std::size_t num, const void * hint = static_cast<const void *>(0)) {
            return static_cast<pointer>( malloc(sizeof(value_type) * num));
        }
        
        void deallocate(pointer p, std::size_t num [[maybe_unused]]) {
            free(p);
        }
        
        static
        pointer staticAllocate(std::size_t num, const void * hint = 0) {
            return static_cast<pointer>( malloc(sizeof(value_type) * num));
        }
        
        static
        void staticDeallocate(pointer p, std::size_t num [[maybe_unused]]) {
            free(p);
        }
        
        static
        void staticDeallocate(void * p, std::size_t num [[maybe_unused]]) {
            free(p);
        }
        
        
    };
    
} /// namespace morphstore
#endif //MORPHSTORE_INCLUDE_CORE_MEMORY_COLUMNALLOCATOR_H
