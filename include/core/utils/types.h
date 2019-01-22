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
