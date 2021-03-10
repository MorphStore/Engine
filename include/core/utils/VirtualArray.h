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


#ifndef MORPHSTORE_INCLUDE_CORE_UTILS_VIRTUALARRAY_H
#define MORPHSTORE_INCLUDE_CORE_UTILS_VIRTUALARRAY_H

#include <stdlibs>

#include <interfaces>

namespace morphstore {
    
    template<typename Type>
    class VirtualArray {
      Type value;
      public:
        
        VirtualArray(Type value) : value(value) {};
        
        Type operator[](size_t){
            return value;
        }
        
        Type operator+(Type other){
            return value + other;
        }
        
        VirtualArray<Type>& operator=(Type value_){
            value = value_;
            return *this;
        }
        
        operator Type(){
            return value;
        }
        
        template<IArithmetic TNewType>
        operator TNewType(){
            return value;
        }
    };
}

namespace std {
    template<typename T>
    struct is_arithmetic<morphstore::VirtualArray<T>> : is_arithmetic<T> {};
    template<typename T>
    struct is_arithmetic<morphstore::VirtualArray<T>*> : is_arithmetic<T> {};
}

#endif //MORPHSTORE_INCLUDE_CORE_UTILS_VIRTUALARRAY_H
