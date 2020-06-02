/**********************************************************************************************
 * Copyright (C) 2020 by MorphStore-Team                                                      *
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
 * @file append.h
 * @brief The interface of append operator
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_APPEND_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_APPEND_H

#include <core/storage/replicated_column.h>
#include <core/storage/replicated_write_iterator.h>

namespace morphstore {

/**
 * Append operator. Appends given value to the replicated column.
 *
 * Example:
 * - inOriginalCol:    [ 4,   3,  1,   0,   2,   5]
 * - value:            [ 10]
 * - num:              [ 1 ]
 * - InOriginalCol:    [ 4,   3,  1,   0,   2,   5,  10]
 *
 * @param inOriginalCol A column to which value is appended.
 * @param value A value to be appended at the end of inOriginalCol.
 * @param num A number of value elements to be appended at the end of inOriginalCol.
 * @return inOriginalCol exteded by values
 */

template<class t_vector_extension>
void append(replicated_column* const inOriginalCol,
       const size_t value, const size_t num
);

}
#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_APPEND_H