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
 * @file ostream.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_STL_WRAPPER_OSTREAM_H
#define MORPHSTORE_CORE_MEMORY_STL_WRAPPER_OSTREAM_H

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_ALLOCATORS_GLOBAL_SCOPE_ALLOCATOR_H
#  error "Perpetual (global scoped) allocator ( allocators/global_scope_allocator.h ) has to be included before all stl_wrapper."
#endif

#include <sstream>
#include <fstream>
#include <unistd.h>

namespace morphstore {

using ostring_stream = std::basic_ostringstream< char, std::char_traits< char >, global_scope_stdlib_allocator< char > >;
using basic_stringbuf = std::basic_stringbuf< char, std::char_traits< char >, global_scope_stdlib_allocator< char > >;
using basic_filebuf = std::basic_filebuf< char, std::char_traits< char > >;

class outbuf : public morphstore::basic_stringbuf {
   protected:
      int m_FD;
   public:
      outbuf( int p_FD ) : m_FD( p_FD ) { }
   protected:
      virtual int_type overflow( int_type p_Character ) {
         if( p_Character != EOF ) {
            char z = p_Character;
            if( write( m_FD, &z, 1 ) != 1 ) {
               return EOF;
            }
         }
         return p_Character;
      }
      virtual std::streamsize xsputn( const char * p_Text, std::streamsize p_Num ) {
         return write( m_FD, p_Text, p_Num );
      }
};

class ostream : public std::ostream {
   protected:
      outbuf m_Buffer;
   public:
      ostream( ) : std::ostream( 0 ), m_Buffer( 1 ) {
         rdbuf( &m_Buffer );
      }
};

}


#endif //MORPHSTORE_CORE_MEMORY_STL_WRAPPER_OSTREAM_H
