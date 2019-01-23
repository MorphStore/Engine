/**********************************************************************************************
 * Copyright (C) 2019 by Johannes Pietrzyk                                                    *
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


/**
 * @file datatypes.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_UTILS_TYPES_H
#define MORPHSTORE_CORE_UTILS_TYPES_H

#include <cstddef>

namespace morphstore {

using size_t = std::size_t;

constexpr std::size_t operator""_B( unsigned long long v ) {
   return v;
}
constexpr std::size_t operator""_KB( unsigned long long v ) {
   return ( 1024 * v );
}
constexpr std::size_t operator""_MB( unsigned long long v ) {
   return ( 1024 * 1024 * v );
}

constexpr std::size_t operator""_GB( unsigned long long v ) {
   return ( 1024 * 1024 * 1024 * v );
}


}
#endif //MORPHSTORE_CORE_UTILS_TYPES_H
