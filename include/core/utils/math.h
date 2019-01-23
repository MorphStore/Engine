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
 * @file math.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_UTILS_MATH_H
#define MORPHSTORE_CORE_UTILS_MATH_H

#include "types.h"

#include <cstdint>

namespace morphstore {

constexpr bool is_power_of_two( size_t pN ) {
   if( pN == 0 )
      return false;
   if( ( pN & ( pN - 1 ) ) == 0 )
      return true;
   return false;
}

constexpr std::size_t to_the_power_of_two( uint8_t N ) {
   if( N == 0 )
      return 1;
   if( N == 1 )
      return 2;
   return 1 << N;
}

constexpr std::size_t log2( size_t n ) {
   return ( ( n < 2 ) ? 1 : 1 + log2( n / 2 ) );
}

}

#endif //MORPHSTORE_CORE_UTILS_MATH_H
