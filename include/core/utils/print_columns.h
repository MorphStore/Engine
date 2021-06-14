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
//#include <storage>
#include "core/storage/column.h"

namespace morphstore {
    
    template< typename first, typename ... last >
    static
    size_t get_max_size(first f, last ... l) {
        if constexpr(sizeof...(last) > 1) {
            return std::max(f->get_count_values(), get_max_size(l...));
        } else {
            return f->get_count_values();
        }
    }
    
    template< uint64_t width = 5 >
    static
    std::string print_element(size_t index, const column <uncompr_f> * col) {
        std::stringstream out;
        out << std::setw(width);
        if (index < col->get_count_values()) {
            out << ((uint64_t *) col->get_data())[index] << " | ";
        } else {
            out << " " << " | ";
        }
        return out.str();
    }
    
    template< uint64_t width = 5, IStoragePtr...TCols >
    static
    void print_columns(uint64_t offset, TCols...cols) {
//        for (size_t i = 0; i < get_max_size(cols...); ++ i) {
//            std::cout << std::setw(4) << offset << "/" << std::setw(4) << i << ": ";
//            (print_element<width>(i, cols), ...);
//            std::cout << std::endl;
//            ++offset;
//        }
        std::stringstream out;
        
        /// header
        out << "Line:";
        for(size_t i = 0; i < sizeof...(TCols); ++i){
            out << " col" << std::setw(width - 3) << i << " |";
        }
        out << "\n";
        
        /// data
        for (size_t i = 0; i < get_max_size(cols...); ++ i) {
            /// line number
            out << std::setw(4) << (i+offset) << ": ";
            /// print n-th row of each column
            (out << ... << print_element<width>(i, cols));
            out << "\n";
        }
        /// print to cout
        std::cout << out.str() << std::flush;
    }
    
    template< uint64_t width = 5, IStoragePtr...TCols >
    static
    void print_columns(TCols...cols) {
        std::stringstream out;
        
        /// header
        out << "Line:";
        for(size_t i = 0; i < sizeof...(TCols); ++i){
            out << " col" << std::setw(width - 3) << i << " |";
        }
        out << "\n";
        
        /// data
        for (size_t i = 0; i < get_max_size(cols...); ++ i) {
            /// line number
            out << std::setw(4) << i << ": ";
            /// print n-th row of each column
            (out << ... << print_element<width>(i, cols));
            out << "\n";
        }
        /// print to cout
        std::cout << out.str() << std::flush;
    }
}

#endif //MORPHSTORE_INCLUDE_CORE_UTILS_PRINT_COLUMNS_H
