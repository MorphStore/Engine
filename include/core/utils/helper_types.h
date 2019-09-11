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
 * @file helper_types.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_UTILS_HELPER_TYPES_H
#define MORPHSTORE_CORE_UTILS_HELPER_TYPES_H

#if !defined(MSV_NO_LOG) && defined(DEBUG)
#  include <typeinfo>
#endif
#include <type_traits>
namespace morphstore {

   struct voidptr_helper {
      void const * const m_Ptr;
      constexpr voidptr_helper(void const * const p_Ptr): m_Ptr{p_Ptr}{}
      template<typename T>
      MSV_CXX_ATTRIBUTE_INLINE MSV_CXX_ATTRIBUTE_CONST
      operator T const * () const {
         trace("[ VoidPtr Helper ] Implicit cast [ const void * const ] --> [ const ", typeid(T).name(), " * ]");
         return reinterpret_cast<T const * const >( m_Ptr );
      }
      template<typename T>
      MSV_CXX_ATTRIBUTE_INLINE MSV_CXX_ATTRIBUTE_CONST
      operator T * () const {
         trace("[ VoidPtr Helper ] Implicit cast [ const void * const ] --> [ ", typeid(T).name(), " * ]");
         return reinterpret_cast<T * >( const_cast< void * >( m_Ptr ) );
      }
   };
   using voidptr_t = voidptr_helper;

   template< int Alignment, typename T, size_t N >
   struct alignas( Alignment ) aligned_array : public std::array< T, N >{
      template< class... U >
      aligned_array( U&&... u ) : std::array< T, N >{ std::forward< U >( u )... } {}
   };

#  define IMM_INT32(N) std::integral_constant< int32_t, N >()
#  define IMM_UINT32(N) std::integral_constant< uint32_t, N >()
#  define IMM_INT64(N) std::integral_constant< int64_t, N >()
#  define IMM_UINT64(N) std::integral_constant< uint64_t, N >()

}
#endif //MORPHSTORE_CORE_UTILS_BASIC_TYPES_H
