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


#ifndef MORPHSTORE_INCLUDE_CORE_UTILS_PRINT_COLUMNS_H
#define MORPHSTORE_INCLUDE_CORE_UTILS_PRINT_COLUMNS_H

#include <stdlibs>
#include <storage>

namespace morphstore {
    
    template< typename first, typename ... last >
    size_t get_max_size(first f, last ... l) {
        if constexpr(sizeof...(last) > 1) {
            return std::max(f->get_count_values(), get_max_size(l...));
        } else {
            return f->get_count_values();
        }
    }
    
    template< uint64_t width = 5 >
    void print_element(size_t index, const column <uncompr_f> * col) {
        std::cout << std::setw(width);
        if (index < col->get_count_values()) {
            std::cout << ((uint64_t *) col->get_data())[index] << " | ";
        } else {
            std::cout << " " << " | ";
        }
    }
    
    template< uint64_t width = 5, typename...TCols >
    void print_columns(uint64_t& offset, TCols...cols) {
        for (size_t i = 0; i < get_max_size(cols...); ++ i) {
            std::cout << std::setw(4) << offset << "/" << std::setw(4) << i << ": ";
            (print_element<width>(i, cols), ...);
            std::cout << std::endl;
            ++offset;
        }
    }
    
    template< uint64_t width = 5, typename...TCols >
    void print_columns(TCols...cols) {
        for (size_t i = 0; i < get_max_size(cols...); ++ i) {
            std::cout << std::setw(4) << i << ": ";
            (print_element<width>(i, cols), ...);
            std::cout << std::endl;
        }
    }
    
}

#endif //MORPHSTORE_INCLUDE_CORE_UTILS_PRINT_COLUMNS_H
