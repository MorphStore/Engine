/**********************************************************************************************
 * Copyright (C) 2019 by Johannes Pietrzyk                                                    *
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
 * @file preprocessor.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_UTILS_PREPROCESSOR_H
#define MORPHSTORE_CORE_UTILS_PREPROCESSOR_H



#ifndef MSV_CXX_ATTRIBUTE_PPUNUSED
#  if defined(__clang__) || defined(__GNUC__)
#     define MSV_CXX_ATTRIBUTE_PPUNUSED __attribute__((unused))
#  endif
#endif

#ifndef MSV_CXX_ATTRIBUTE_MALLOC
#  if defined(__clang__) || defined(__GNUC__)
#     define MSV_CXX_ATTRIBUTE_MALLOC __attribute__((malloc))
#  endif
#endif

#ifndef MSV_CXX_ATTRIBUTE_ALLOC_SIZE
#  if defined(__clang__) || defined(__GNUC__)
#     define MSV_CXX_ATTRIBUTE_ALLOC_SIZE(X) __attribute__((alloc_size(X)))
#  endif
#endif

#ifndef MSV_CXX_ATTRIBUTE_FORCE_INLINE
#  if defined(__clang__) || defined(__GNUC__)
#     define MSV_CXX_ATTRIBUTE_FORCE_INLINE inline __attribute__((always_inline))
#  endif
#endif

#ifndef MSV_CXX_ATTRIBUTE_INLINE
#  if defined(__clang__) || defined(__GNUC__)
#     define MSV_CXX_ATTRIBUTE_INLINE inline
#  endif
#endif

#ifndef MSV_CXX_ATTRIBUTE_COLD
#  if defined(__clang__) || defined(__GNUC__)
#     define MSV_CXX_ATTRIBUTE_COLD __attribute__((cold))
#  endif
#endif

#ifndef MSV_CXX_ATTRIBUTE_HOT
#  if defined(__clang__) || defined(__GNUC__)
#     define MSV_CXX_ATTRIBUTE_HOT __attribute__((hot))
#  endif
#endif

#ifndef MSV_CXX_ATTRIBUTE_PURE
#  if defined(__clang__) || defined(__GNUC__)
#     define MSV_CXX_ATTRIBUTE_PURE __attribute__((pure))
#  endif
#endif

#ifndef MSV_CXX_ATTRIBUTE_CONST
#  if defined(__clang__) || defined(__GNUC__)
#     define MSV_CXX_ATTRIBUTE_CONST __attribute__((const))
#  endif
#endif

#ifndef __THROW
#  define __THROW
#endif

#endif //MORPHSTORE_CORE_UTILS_PREPROCESSOR_H
