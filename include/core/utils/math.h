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
