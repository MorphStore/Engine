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
 */

#ifndef MORPHSTORE_CORE_MORPHING_FORMAT_H
#define MORPHSTORE_CORE_MORPHING_FORMAT_H

#include <core/utils/math.h>
#include <core/utils/basic_types.h>
#include <vector/general_vector.h>

#include <cstdint>

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
     * Assumes that the provided number of data elements is a multiple of
     * `m_BlockSize`.
     * 
     * @param p_CountValues The number of data elements.
     * @return The maximum size (in bytes) that could be required in this
     * format.
     */
    static size_t get_size_max_byte(size_t p_CountValues) = delete;

    static const size_t m_BlockSize;
};

/**
 * @brief The uncompressed format, i.e., a sequence of 64-bit integers.
 */
// @todo This should be moved to "uncompr.h", but since it is used at so many
// places, we should leave this for later.
struct uncompr_f : public format {
    static size_t get_size_max_byte(size_t p_CountValues) {
        return convert_size<uint64_t, uint8_t>(p_CountValues);
    }
    
    static const size_t m_BlockSize = 1;
};

template<class t_format>
class read_iterator;

template<
        class t_vector_extension,
        class t_format,
        template<class /*t_vector_extension*/> class t_op_processing_unit
>
struct decompress_and_process_batch {
    static void apply(
            const uint8_t * & p_In8,
            size_t p_CountIn8,
            typename t_op_processing_unit<t_vector_extension>::state_t & p_State
    );
};

template<class t_vector_extension, class t_format>
struct write_iterator {
    IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
            
    write_iterator(uint8_t * p_Out);
    void write(vector_t p_Data, vector_mask_t p_Mask);
    void done();
    size_t get_count() const;
};

}
#endif //MORPHSTORE_CORE_MORPHING_FORMAT_H
