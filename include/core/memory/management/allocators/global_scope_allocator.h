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
 * @file global_scope_allocator.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_ALLOCATORS_GLOBAL_SCOPE_ALLOCATOR_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_ALLOCATORS_GLOBAL_SCOPE_ALLOCATOR_H


#ifndef MORPHSTORE_CORE_MEMORY_GLOBAL_MM_HOOKS_H
#  error "Memory Hooks ( global/mm_hooks.h ) has to be included before perpetual (global scoped) allocator."
#endif

#include <core/utils/basic_types.h>
#include <core/utils/preprocessor.h>
#include <core/memory/global/mm_hooks.h>

#include <limits>
//#include <cstdlib>
#include <utility>

namespace morphstore {
   template< class C >
   class global_scope_stdlib_allocator {
   public:
      typedef C               value_type;
      typedef C *             pointer;
      typedef C const *       const_pointer;
      typedef C &             reference;
      typedef C const &       const_reference;
      typedef size_t     size_type;
      typedef ptrdiff_t  difference_type;
   private:
      static constexpr size_type m_MaxSize = std::numeric_limits< size_t >::max( ) / sizeof( C );
   public:
      template< class U >
      struct rebind {
         typedef global_scope_stdlib_allocator< U > other;
      };

      pointer address( reference value ) const {
         return &value;
      }
      const_pointer address( const_reference value ) const {
         return &value;
      }

      global_scope_stdlib_allocator( void ) throw( ) {
         if( ! (
            ( stdlib_malloc_ptr == nullptr ) ||
            ( stdlib_malloc_ptr == nullptr ) ||
            ( stdlib_malloc_ptr == nullptr ) )
         )
            if( ! init_mem_hooks( ) ) {
//               exit( 1 );
               throw( 1 );
            }
      }
      global_scope_stdlib_allocator( const global_scope_stdlib_allocator & ) throw( ) { }
      template< class U >
      global_scope_stdlib_allocator( const global_scope_stdlib_allocator< U >& ) throw( ) { }
      ~global_scope_stdlib_allocator( ) throw( ) { }

      size_type max_size( ) const throw( ) {
         return m_MaxSize;
      }

      pointer allocate( size_t p_AllocCount, const void * = 0 ) {
         return static_cast< pointer >( stdlib_malloc( p_AllocCount * sizeof( C )  ) );
      }

      void construct( pointer p_Ptr, const C & value ) {
         new( static_cast< void * >( p_Ptr ) ) C( value );
      }

      template< class... Args >
      void construct( pointer p_Ptr, Args&& ... args ) {
         new( static_cast< void * >( p_Ptr ) ) C( std::forward< Args >( args ) ... );
      }

      void destroy( pointer p_FreePtr ) {
         p_FreePtr->~C( );
      }

      void deallocate( pointer p_FreePtr, MSV_CXX_ATTRIBUTE_PPUNUSED size_t p_NumElements ) {
          stdlib_free( static_cast< void * >( p_FreePtr ) );
      }
   };
   template <class T1, class T2>
   bool operator== (const global_scope_stdlib_allocator<T1>&,
                    const global_scope_stdlib_allocator<T2>&) throw() {
      return true;
   }
   template <class T1, class T2>
   bool operator!= (const global_scope_stdlib_allocator<T1>&,
                    const global_scope_stdlib_allocator<T2>&) throw() {
      return false;
   }


}

#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_ALLOCATORS_GLOBAL_SCOPE_ALLOCATOR_H
