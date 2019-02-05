/**
 * @file helper.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_UTILS_HELPER_H
#define MORPHSTORE_CORE_UTILS_HELPER_H

#include <cstddef>
namespace morphstore {

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

   template< typename B, typename T >
   inline bool instanceof( T const * p_Ptr ) {
      return dynamic_cast< B const * >( p_Ptr ) != nullptr;
   }


}
#endif //MORPHSTORE_CORE_UTILS_HELPER_H
