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
 * @file basic_operators.h
 * @brief A whole-column basic operators on uncompressed data
 * @todo Currently, the iterator works only with scalars. When this changes, we
 * should move this file to another directory.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_BASIC_OPERATORS_H
#define MORPHSTORE_CORE_OPERATORS_BASIC_OPERATORS_H

#include <core/operators/interfaces/basic_operators.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace morphstore {

template<>
const column<uncompr_f> *
sequential_read<vectorlib::scalar<vectorlib::v64<uint64_t>>>(
        const column<uncompr_f> * const inDataCol
) {
    const size_t inDataCount = inDataCol->get_count_values();
    const uint64_t * const inData = inDataCol->get_data();

    // Exact allocation size (for uncompressed data).
    auto outDataCol = new column<uncompr_f>(sizeof(uint64_t));
    uint64_t * const outData = outDataCol->get_data();

    *outData = 0;
    for(unsigned i = 0; i < inDataCount; i++)
        *outData += inData[i];

    outDataCol->set_meta_data(1, sizeof(uint64_t));

    return outDataCol;
}

template<>
const column<uncompr_f> *
random_read<vectorlib::scalar<vectorlib::v64<uint64_t>>>(
        const column<uncompr_f> * const inAccessPos,
        const column<uncompr_f> * const inDataCol
) {
    const size_t inDataCount = inDataCol->get_count_values();

    if(inDataCount != inAccessPos->get_count_values())
        throw std::runtime_error(
                "random_read: inAccessPos and inDataCol must contain the same number "
                "of data elements"
        );

    const uint64_t * const inAccess = inAccessPos->get_data();
    const uint64_t * const inData = inDataCol->get_data();

    // Exact allocation size (for uncompressed data).
    auto outDataCol = new column<uncompr_f>(sizeof(uint64_t));
    uint64_t * const outData = outDataCol->get_data();

    *outData = 0;
    for(unsigned i = 0; i < inDataCount; i++)
        *outData += inData[inAccess[i]];

    outDataCol->set_meta_data(1, sizeof(uint64_t));

    return outDataCol;
}

template<>
const column<uncompr_f> *
sequential_write<vectorlib::scalar<vectorlib::v64<uint64_t>>>(
        column<uncompr_f> * const inDataCol
) {
    const size_t inDataCount = inDataCol->get_count_values();
    uint64_t * const inData = inDataCol->get_data();

    for(unsigned i = 0; i < inDataCount; i++)
        inData[i] = i;

    inDataCol->set_meta_data(inDataCount, sizeof(uint64_t));

    return inDataCol;
}

template<>
const column<uncompr_f> *
random_write<vectorlib::scalar<vectorlib::v64<uint64_t>>>(
        const column<uncompr_f> * const inAccessPos,
        column<uncompr_f> * const inDataCol
) {
    const size_t inDataCount = inDataCol->get_count_values();

    if(inDataCount != inAccessPos->get_count_values())
        throw std::runtime_error(
                "random_write: inAccessPos and inDataCol must contain the same number "
                "of data elements"
        );

    const uint64_t * const inAccess = inAccessPos->get_data();
    uint64_t * const inData = inDataCol->get_data();

    for(unsigned i = 0; i < inDataCount; i++)
        inData[inAccess[i]] = i;

    inDataCol->set_meta_data(inDataCount, sizeof(uint64_t));

    return inDataCol;
}

template<>
column<uncompr_f> *
append_chunk<vectorlib::scalar<vectorlib::v64<uint64_t>>>(
        column<uncompr_f> * const inOriginalCol,
        const column<uncompr_f> * const inChunkCol
) {
    const size_t resultSize = inOriginalCol->get_count_values() + inChunkCol->get_count_values();
    const size_t originalEnd = inOriginalCol->get_count_values();

    if(resultSize > inOriginalCol->get_count_values())
        throw std::runtime_error(
                "append_chunk: out of preallocated memory"
        );

    uint64_t * const inOriginal = inOriginalCol->get_data();
    const uint64_t * const inChunk = inChunkCol->get_data();

    for(unsigned i = 0; i < inChunkCol->get_count_values(); i++)
        inOriginal[originalEnd + i] = inChunk[i];

    inOriginalCol->set_meta_data(resultSize, sizeof(uint64_t));

    return inOriginalCol;
}

}
#endif // MORPHSTORE_CORE_OPERATORS_BASIC_OPERATORS_H