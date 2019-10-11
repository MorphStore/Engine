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
#include <core/utils/preprocessor.h>

#include <limits>

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
MSV_CXX_ATTRIBUTE_INLINE constexpr unsigned round_up_div( unsigned numerator, unsigned denominator ) {
    return ( numerator + denominator - 1 ) / denominator;
}

MSV_CXX_ATTRIBUTE_INLINE constexpr unsigned round_up_to_multiple(
        size_t p_Size, size_t p_Factor
) {
    return round_up_div(p_Size, p_Factor) * p_Factor;
}

MSV_CXX_ATTRIBUTE_INLINE constexpr unsigned round_down_to_multiple(
        size_t p_Size, size_t p_Factor
) {
    return p_Size / p_Factor * p_Factor;
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
constexpr inline size_t convert_size(size_t p_SizeInSrcUnits) {
    return p_SizeInSrcUnits * sizeof(t_src) / sizeof(t_dst);
}

/**
 * @brief Calculates the number of effective bits of the given 64-bit value.
 * 
 * This is the minimum number of bits required to represent this value. Note
 * that, by definition, the integer `0` has one effective bit. Thus, this
 * function's return value is always in the range [1, 64].
 * 
 * @param p_Val The 64-bit value.
 * @return The number of bits required to represent the given value.
 */
constexpr inline unsigned effective_bitwidth(uint64_t p_Val) {
    // Should we ever change the definition such that 0 has 0 effective bits,
    // then don't forget that the return value of __builtin_clzll is undefined
    // for 0.
    return std::numeric_limits<uint64_t>::digits - __builtin_clzll(p_Val | 1);
}

constexpr inline unsigned zero_bytes(uint64_t p_Val) {
    // Should we ever change the definition such that 0 has 0 effective bits,
    // then don't forget that the return value of __builtin_clzll is undefined
    // for 0.
    return __builtin_clzll(p_Val | 1) / bitsPerByte;
}

/**
 * @brief Calculates the number of effective bytes of the given 64-bit value.
 * 
 * This is the minimum number of bytes required to represent this value. Note
 * that, by definition, the integer `0` has one effective byte. Thus, this
 * function's return value is always in the range [1, 8].
 * 
 * @param p_Val The 64-bit value.
 * @return The number of bytes required to represent the given value.
 */
constexpr inline unsigned effective_bytewidth(uint64_t p_Val) {
    return sizeof(uint64_t) - zero_bytes(p_Val);
}

/**
 * @brief Calculates the maximum unsigned integer of the given bit width.
 * 
 * The template parameter `t_uintX_t` should be one of the `uint*_t` types from
 * the header `&lt;cstdint&gt;`.
 * 
 * @param p_Bw The bit width.
 * @return The highest unsigned integer of the given bit width.
 */
template<typename t_uintX_t>
constexpr inline t_uintX_t bitwidth_max(unsigned p_Bw) {
    // The special case for the maximum bit width is necessary since it is
    // (unfortunately) not allowed to left-shift an integer by the number of
    // its digits.
    return (p_Bw == std::numeric_limits<t_uintX_t>::digits)
            ? std::numeric_limits<t_uintX_t>::max()
            : (static_cast<t_uintX_t>(1) << p_Bw) - 1;
}

/**
 * @brief Calculates the bit mask that can be used for replacing a modulo
 * operation by a bitwise AND.
 * 
 * `a % b` can be replaced by `a & mask_for_mod_power_of_two(b)`, if `b` is a
 * power of two.
 * 
 * @param p_Divisor Must be a power of two.
 * @return A bit mask for replacing modulo by bitwise AND.
 */
constexpr inline uint64_t mask_for_mod(uint64_t p_Divisor) {
    return p_Divisor - 1;
}

/**
 * Calculates the shift offset that can be used for replacing an integer
 * multiplication(division) by a left(right) shift.
 * 
 * `a * b` can be replaced by `a << shift_for_muldiv_power_of_two(b)` and
 * `a / b` can be replaced by `a >> shift_for_muldiv_power_of_two(b)`, if `b`
 * is a power of two.
 * 
 * @param p_FactorOrDivisor Must be a power of two.
 * @return A shift offset for replacing multiplicatio(divison) by left(right)
 * shift.
 */
constexpr inline uint64_t shift_for_muldiv(uint64_t p_FactorOrDivisor) {
    return (p_FactorOrDivisor == 1)
        ? 0
        : effective_bitwidth(p_FactorOrDivisor - 1);
}

}

#endif //MORPHSTORE_CORE_UTILS_MATH_H
