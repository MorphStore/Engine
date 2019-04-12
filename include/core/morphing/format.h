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
 * @file format.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MORPHING_FORMAT_H
#define MORPHSTORE_CORE_MORPHING_FORMAT_H

#include <core/utils/math.h>

namespace morphstore {

/**
 * @brief The base class of all format implementations.
 */
struct format {
    /**
     * @brief Provides a pessimistic estimation of the maximum possible size a
     * column containing the given number of data elements could have when
     * represented in this format.
     * 
     * This size can be used for determining the number of bytes that must be
     * allocated for a column. To prevent buffer overflows in all cases, it is
     * very important not to underestimate this size.
     * 
     * @param p_CountValues The number of data elements.
     * @return The maximum size (in bytes) that could be required in this
     * format.
     */
    static size_t get_size_max_byte(size_t p_CountValues) = delete;
};

/**
 * @brief The uncompressed format, i.e., a sequence of 64-bit integers.
 */
struct uncompr_f : public format {
    static size_t get_size_max_byte(size_t p_CountValues) {
        return convert_size<uint64_t, uint8_t>(p_CountValues);
    }
};

}
#endif //MORPHSTORE_CORE_MORPHING_FORMAT_H
