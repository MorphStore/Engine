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


#ifndef MORPHSTORE_STORAGE_INTERFACES_H
#define MORPHSTORE_STORAGE_INTERFACES_H

#include <abridge/stdlibs>

namespace morphstore {
    /// Forward declaration of class Storage in <core/storage/Storage.h>
    class Storage;
    
    template<typename TStorage>
    struct is_storage {
        static constexpr bool value = std::is_base_of<Storage, typename std::remove_const<TStorage>::type>::value;
    };
    template<typename TStorage>
    struct is_storage_ptr {
        static constexpr bool value = std::is_pointer<TStorage>::value and is_storage<typename std::remove_pointer<TStorage>::type>::value;
    };
    
    
    #ifdef USE_CPP20_CONCEPTS
    
        template<class TStorage>
        concept IStorage = is_storage<TStorage>::value;
        template<class TStorage>
        concept IStoragePtr = is_storage_ptr<TStorage>::value;
//
        
    #else
      
        #define IStorage typename
      
    #endif
    
    
    
    
    template<typename TArithmeitc>
    struct is_arithmetic_ptr {
        static constexpr bool value = std::is_pointer<TArithmeitc>::value and std::is_arithmetic<typename std::remove_pointer<TArithmeitc>::type>::value;
    };
    
    #ifdef USE_CPP20_CONCEPTS
    
        template<class TArithmeitc>
        concept IArithmeticPtr = morphstore::is_arithmetic_ptr<TArithmeitc>::value;
        
    #else
      
        #define IArithmeticPtr typename
        
    #endif
    
} // namespace

#endif //MORPHSTORE_STORAGE_INTERFACES_H
