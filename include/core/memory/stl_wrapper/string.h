/**
 * @file string.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_STL_WRAPPER_STRING_H
#define MORPHSTORE_CORE_MEMORY_STL_WRAPPER_STRING_H

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_ALLOCATORS_PERPETUAL_ALLOCATOR_H
#  error "Perpetual allocator ( allocators/perpetual_allocator.h ) has to be included before all stl_wrapper."
#endif

#include <string>

namespace morphstore {

   typedef std::basic_string< char, std::char_traits< char >, perpetual_stdlib_allocator< char > > string;
   typedef std::basic_string< wchar_t, std::char_traits< wchar_t >, perpetual_stdlib_allocator< wchar_t > > wstring;
}
#endif //MORPHSTORE_CORE_MEMORY_STL_WRAPPER_STRING_H
