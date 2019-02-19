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
 * @file string.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_STL_WRAPPER_STRING_H
#define MORPHSTORE_CORE_MEMORY_STL_WRAPPER_STRING_H

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_ALLOCATORS_GLOBAL_SCOPE_ALLOCATOR_H
#  error "Perpetual (global scoped) allocator ( allocators/global_scope_allocator.h ) has to be included before all stl_wrapper."
#endif

#include <string>

namespace morphstore {

   typedef std::basic_string< char, std::char_traits< char >, global_scope_stdlib_allocator< char > > string;
   typedef std::basic_string< wchar_t, std::char_traits< wchar_t >, global_scope_stdlib_allocator< wchar_t > > wstring;
}
#endif //MORPHSTORE_CORE_MEMORY_STL_WRAPPER_STRING_H
