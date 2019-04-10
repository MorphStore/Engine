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

/**
 * @file math.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_UTILS_MATH_H
#define MORPHSTORE_CORE_UTILS_MATH_H

#include <core/utils/basic_types.h>

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
   return 1 << N;
}

constexpr std::size_t log2( size_t n ) {
   return ( ( n < 2 ) ? 1 : 1 + log2( n / 2 ) );
}

/**
 * Returns the quotient of numerator divided by denominator rounded up to the
 * next multiple of denominator.
 * @param numerator
 * @param denominator
 * @return 
 */
inline unsigned round_up_div( unsigned numerator, unsigned denominator ) {
    return ( numerator + denominator - 1 ) / denominator;
}

const size_t bitsPerByte = 8;

/**
 * @brief Converts the size of something counted in units of some source type
 * `t_src` to the size counted in units of some destination type `t_dst`.
 * 
 * For instance, assume `count64` is the number of 64-bit elements in an array.
 * The number of 128-bit vectors in this array can be obtained as follows:
 * 
 *     size_t count128 = convert_size<uint64_t, __m128i>(count64);
 * 
 * Note that, when converting to larger units, any remainder is ignored. For
 * instance, three 64-bit elements are one 128-bit element.
 * 
 * @param p_SizeSrc The size in units of the source type.
 * @return The size in units of the destination type.
 */
template<typename t_src, typename t_dst>
inline size_t convert_size(size_t p_SizeInSrcUnits) {
    return p_SizeInSrcUnits * sizeof(t_src) / sizeof(t_dst);
}

}

#endif //MORPHSTORE_CORE_UTILS_MATH_H
