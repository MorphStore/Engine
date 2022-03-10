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
 * @file leak_detection.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_GLOBAL_LEAK_DETECTION_H
#define MORPHSTORE_CORE_MEMORY_GLOBAL_LEAK_DETECTION_H


#include <iomanip>

namespace morphstore{

class leak_detector {
   struct memory_chunk {
      memory_chunk * m_Next;
      void const * const m_Pointer;
      size_t const m_Size;
      bool m_Freed;
      memory_chunk( memory_chunk * p_Next, void const * const p_Pointer, size_t const p_Size ) :
         m_Next{ p_Next },
         m_Pointer{ p_Pointer },
         m_Size{ p_Size },
         m_Freed{ false } { }

      void print_address( morphstore::ostream & stream ) const {
         stream << " | " << std::setw( 18 ) << m_Pointer << " | ";
      }
      void print_size( morphstore::ostream & stream ) const {
         stream << " | ";
         if( m_Size < 1024 ) {
            stream << std::setw(10) << m_Size << " B";
         } else if( m_Size < 1024 * 1024 ) {
            stream << std::setw(9) << (double)m_Size/(double)1024 << " KB";
         } else if( m_Size < 1024 * 1024 * 1024 ) {
            stream << std::setw(9) << (double)m_Size/(double)(1024 * 1024 ) << " MB";
         } else if( m_Size < ( (size_t)1024 * (size_t)1024 * (size_t)1024 * (size_t)1024 ) )  {
            stream << std::setw(9) << (double)m_Size/(double)( (size_t)1024 * (size_t)1024 * (size_t)1024 ) << " GB";
         }
         stream << "       | ";
      }
      void print_freed( morphstore::ostream & stream ) const {
         if( m_Freed )
            stream << " | " << std::setw( 12 ) << "freed" << "       | ";
         else
            stream << " | " << std::setw( 14 ) << "NOT freed" << "     | ";
      }
   };
   private:
      memory_chunk * m_RootMem;
      memory_chunk * m_TailMem;
      size_t m_AllocateCount;
      size_t m_FreeCount;
      size_t m_FreeSize;
      size_t m_AllocateSize;
      leak_detector( void ): m_AllocateCount{ 0 }, m_FreeCount{ 0 }, m_FreeSize {0}, m_AllocateSize{ 0 } {
         void * rootUninitialized = stdlib_malloc_ptr( sizeof( memory_chunk )  );
         new( rootUninitialized ) memory_chunk(nullptr, nullptr, 0 );
         m_RootMem = reinterpret_cast< memory_chunk * >( rootUninitialized );
         m_TailMem = m_RootMem;
      }
   public:
      static leak_detector & get_instance(void) {
         static leak_detector instance;
         return instance;
      }

      ~leak_detector() {
         get_tracked_memory_access();
         memory_chunk * handle = m_RootMem;
         memory_chunk * nextHandle;
         do {
            nextHandle = handle->m_Next;
            stdlib_free_ptr( handle );
            handle = nextHandle;
         }while( handle != nullptr );
      }
      void malloc_called( void const * const p_Ptr, size_t const p_Size ) {
         void * newVoidElement = stdlib_malloc_ptr( sizeof( memory_chunk )  );
         new( newVoidElement ) memory_chunk( nullptr, p_Ptr, p_Size );
         memory_chunk * newElement = reinterpret_cast< memory_chunk * >( newVoidElement );
         m_TailMem->m_Next = newElement;
         m_TailMem = newElement;
         m_AllocateCount++;
         m_AllocateSize += p_Size;
      }
      void free_called( void const * const p_Ptr ) {
         memory_chunk * handle = m_RootMem->m_Next;
         while( handle != nullptr ) {
            if( handle->m_Pointer == p_Ptr ) {
               if( ! handle->m_Freed ) {
                  handle->m_Freed = true;
                  m_FreeCount++;
                  m_FreeSize += handle->m_Size;
                  return;
               } else {
                  fprintf( stderr, "Double free of pointer %p.\n", p_Ptr);
                  return;
               }
            }
            handle = handle->m_Next;
         }
         fprintf( stderr, "Try to free what was not allocated ( %p ).\n", p_Ptr );
      }

      void get_tracked_memory_access( ) {
         morphstore::ostream stream;
         stream << " " << std::setfill('*') << std::setw( 26 * m_AllocateCount ) << " " << std::setfill(' ') <<"\n";
         stream << "  Memory Summary: " << "\n"
                << "       Allocated: " << m_AllocateCount << " Chunks. ( " << m_AllocateSize << " Bytes ).\n"
                << "     Deallocated: " << m_FreeCount << " Chunks. ( " << m_FreeSize << " Bytes ).\n";
         stream << " " << std::setfill('-') << std::setw( 26 * m_AllocateCount ) << " " << std::setfill(' ') <<"\n";

         memory_chunk * handle = m_RootMem->m_Next;
         while( handle!= nullptr ) {
            handle->print_address(stream);
            stream << "   ";
            handle = handle->m_Next;
         }
         stream << "\n";
         handle = m_RootMem->m_Next;
         while( handle != nullptr ) {
            handle->print_size(stream);
            if( handle->m_Next != nullptr )
               stream << "-->";
            handle = handle->m_Next;
         }
         stream << "\n";
         handle = m_RootMem->m_Next;
         while( handle != nullptr ) {
            handle->print_freed(stream);
            stream << "   ";
            handle = handle->m_Next;
         }
         stream << "\n";
         stream << " " << std::setfill('*') << std::setw( 26 * m_AllocateCount ) << " " << std::setfill(' ') <<"\n";
      }

};


void leak_detector_malloc_called( void * const p_Ptr, size_t const p_Size ) {
   leak_detector::get_instance( ).malloc_called( p_Ptr, p_Size );
}
void leak_detector_free_called( void const * const p_Ptr ) {
   leak_detector::get_instance( ).free_called( p_Ptr );
}

}
#endif //MORPHSTORE_CORE_MEMORY_GLOBAL_LEAK_DETECTION_H
