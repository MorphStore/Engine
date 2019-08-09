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

#include <core/memory/management/utils/alignment_helper.h>
#include <core/utils/math.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <tuple>

#include <cstdint>

namespace morphstore {

// @todo Document the differences between formats and layouts.

/**
 * @brief The base class of `format` and `layout`.
 * 
 * The interface defined here should be implemented by all subclasses of
 * `format` and `layout`.
 */
struct representation {
    /**
     * @brief Provides a pessimistic estimation of the maximum possible size
     * (in byte) a buffer containing the given number of data elements could
     * have when represented in this format.
     * 
     * The number of data elements must be a multiple of `m_BlockSize`. Thus,
     * it should not be used to determine the allocation size for a column. For
     * arbitrary numbers of data elements, see function
     * `morphstore::get_size_max_byte_any_len`.
     * 
     * To prevent buffer overflows in all cases, it is very important not to
     * underestimate this size.
     * 
     * @param p_CountValues The number of data elements, must be a multiple of
     * `m_BlockSize`.
     * @return The maximum size (in bytes) that could be required in this
     * format.
     */
    static size_t get_size_max_byte(size_t p_CountValues) = delete;

    static const size_t m_BlockSize;
};

/**
 * @brief The base class of all format implementations.
 */
struct format : public representation {
    //
};

/**
 * @brief The base class of all layout implementations.
 */
struct layout : public representation {
    //
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

 /**
  * @brief Provides a pessimistic estimation of the maximum possible size (in
  * byte) a column containing the given number of data elements could have when
  * represented in this format.
  * 
  * This function can handle arbitrary numbers of data elements, by taking into
  * account that compressed columns consist of a compressed part and, perhaps,
  * an uncompressed remainder (which is too small to be represented in the
  * respective format).
  * 
  * This size can be used for determining the number of bytes that must be
  * allocated for a column.
  *
  * @param p_CountValues The number of data elements.
  * @return The maximum size (in bytes) that could be required in this format.
  */
template<class t_format>
MSV_CXX_ATTRIBUTE_FORCE_INLINE size_t get_size_max_byte_any_len(
        size_t p_CountValues
) {
    const size_t countValuesCompr = round_down_to_multiple(
            p_CountValues, t_format::m_BlockSize
    );
    const size_t sizeComprByte = t_format::get_size_max_byte(countValuesCompr);
    // We pessimistically assume that an extra t_format::m_BlockSize data
    // elements need to be stored uncompressed. This way, we account for the
    // case that the final number of logical data elements in the column is
    // less than p_CountValues, which could have the consequence that less data
    // elements can be stored compressed.
    const size_t sizeUncomprByte = uncompr_f::get_size_max_byte(
                    p_CountValues - countValuesCompr + t_format::m_BlockSize
    );
    return get_size_with_alignment_padding(sizeComprByte + sizeUncomprByte);
}

template<>
MSV_CXX_ATTRIBUTE_FORCE_INLINE size_t get_size_max_byte_any_len<uncompr_f>(
        size_t p_CountValues
) {
    // The data buffer of an uncompressed column is, of course, not subdivided
    // into a compressed and an uncompressed part.
    return uncompr_f::get_size_max_byte(p_CountValues);
}

template<class t_format>
class read_iterator;

template<
        class t_vector_extension,
        class t_format,
        template<
                class /*t_vector_extension*/, class ... /*t_extra_args*/
        > class t_op_vector,
        class ... t_extra_args
>
struct decompress_and_process_batch {
    static void apply(
            const uint8_t * & p_In8,
            size_t p_CountIn8,
            typename t_op_vector<
                    t_vector_extension, t_extra_args ...
            >::state_t & p_State
    );
};

// @todo If a write-iterator knows that its data is sorted then it could do
// more optimizations, e.g., when determining the bit width, only the last
// value in the block would need to be considered, since it is always the
// greatest.

/**
 * @brief The interface for writing compressed data selectively.
 */
template<class t_vector_extension, class t_format>
struct selective_write_iterator {
    IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
            
    selective_write_iterator(uint8_t * p_Out);
    
    /**
     * @brief Stores the elements of the given data vector selected by the
     * given mask to the output.
     * 
     * Internally, buffering may take place, such that it is not guaranteed
     * that the data is stored to the output immediately.
     * 
     * `done` must be called after the last call to this function to
     * guarantee that the data is stores to the output in any case.
     * 
     * @param p_Data
     * @param p_Mask
     */
    MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(
            vector_t p_Data, vector_mask_t p_Mask
    );

    MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(
            vector_t p_Data, vector_mask_t p_Mask, uint8_t p_MaskPopCount
    );
    
    /**
     * @brief Makes sure that all possibly buffered data is stored to the
     * output and returns useful information for further processing.
     * 
     * This function should always be called after the last call to `write`.
     * 
     * For compressed output formats, the output's uncompressed rest part is
     * initialized if the number of data elements stored using this instance is
     * not compressible in the output format.
     * 
     * @return A tuple with the following elements:
     * 1. The size of the output's *compressed* part in bytes.
     * 2. `true` if the output's *uncompressed* part has been initialized,
     *    `false` otherwise.
     * 3. A pointer to the end of the stored (un)compressed data, which can be
     *    used to continue storing data to the output.
     */
    std::tuple<size_t, bool, uint8_t *> done();
    
    /**
     * @brief Returns the number of logical data elements that were stored
     * using this instance.
     * @return The number of logical data elements stored using this instance.
     */
    size_t get_count_values() const;
};

/**
 * The interface for writing compressed data non-selectively.
 */
// Currently, we have no implementations for non-selective write-iterators yet.
// Therefore, we delegate to the selective counterpart. This enables the
// implementation of non-selective operators against the non-selective
// interface.
// @todo Implement the non-selective write-iterators.
template<class t_vector_extension, class t_format>
class nonselective_write_iterator {
    IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
    
    selective_write_iterator<t_vector_extension, t_format> m_Wit;
    
public:
    nonselective_write_iterator(uint8_t * p_Out) : m_Wit(p_Out) {
        //
    };
    
    /**
     * @brief Stores the given data vector to the output.
     * 
     * Internally, buffering may take place, such that it is not guaranteed
     * that the data is stored to the output immediately.
     * 
     * `done` must be called after the last call to this function to
     * guarantee that the data is stores to the output in any case.
     * 
     * @param p_Data
     */
    MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(vector_t p_Data) {
        m_Wit.write(
                p_Data,
                bitwidth_max<vector_mask_t>(vector_element_count::value)
        );
    }
    
    /**
     * @brief Makes sure that all possibly buffered data is stored to the
     * output and returns useful information for further processing.
     * 
     * This function should always be called after the last call to `write`.
     * 
     * For compressed output formats, the output's uncompressed rest part is
     * initialized if the number of data elements stored using this instance is
     * not compressible in the output format.
     * 
     * @return A tuple with the following elements:
     * 1. The size of the output's *compressed* part in bytes.
     * 2. `true` if the output's *uncompressed* part has been initialized,
     *    `false` otherwise.
     * 3. A pointer to the end of the stored (un)compressed data, which can be
     *    used to continue storing data to the output.
     */
    std::tuple<size_t, bool, uint8_t *> done() {
        return m_Wit.done();
    }
};

template<class t_vector_extension, class t_format>
struct random_read_access {
    IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
    
    random_read_access(const base_t * p_Data);
    
    MSV_CXX_ATTRIBUTE_FORCE_INLINE vector_t get(const vector_t & p_Positions);
};

}
#endif //MORPHSTORE_CORE_MORPHING_FORMAT_H
