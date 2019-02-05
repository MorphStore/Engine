/**
 * @file ostream.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_STL_WRAPPER_OSTREAM_H
#define MORPHSTORE_CORE_MEMORY_STL_WRAPPER_OSTREAM_H

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_ALLOCATORS_PERPETUAL_ALLOCATOR_H
#  error "Perpetual allocator ( allocators/perpetual_allocator.h ) has to be included before all stl_wrapper."
#endif

#include <sstream>
#include <fstream>
#include <unistd.h>

namespace morphstore {

using ostring_stream = std::basic_ostringstream< char, std::char_traits< char >, memory::perpetual_stdlib_allocator< char > >;
using basic_stringbuf = std::basic_stringbuf< char, std::char_traits< char >, memory::perpetual_stdlib_allocator< char > >;
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
