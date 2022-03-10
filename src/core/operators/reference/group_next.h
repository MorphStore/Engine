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
 * @file group_next.h
 * @brief Template specialization of the group_next-operator for uncompressed
 * inputs and outputs using the scalar processing style. Note that this is a
 * simple reference implementations not tailored for efficiency.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_REFERENCE_GROUP_NEXT_H
#define MORPHSTORE_CORE_OPERATORS_REFERENCE_GROUP_NEXT_H

#include <core/operators/interfaces/group_next.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#ifndef MSV_NO_SELFMANAGED_MEMORY
#include <core/memory/management/allocators/global_scope_allocator.h>
#endif

#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace morphstore {
    
template<>
struct group_next_t<
        vectorlib::scalar<vectorlib::v64<uint64_t>>,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
> {
    static
    const std::tuple<
            const column<uncompr_f> *,
            const column<uncompr_f> *
    >
    apply(
            const column<uncompr_f> * const inGrCol,
            const column<uncompr_f> * const inDataCol,
            const size_t outExtCountEstimate
    ) {
        // This implementation of the binary group-operator covers also the unary
        // case if inGrCol == nullptr .

        const size_t inDataCount = inDataCol->get_count_values();

        if(inGrCol != nullptr && inDataCount != inGrCol->get_count_values())
            throw std::runtime_error(
                    "binary group: inGrCol and inDataCol must contain the same "
                    "number of data elements"
            );

        const uint64_t * const inGr = (inGrCol == nullptr)
                ? nullptr
                : inGrCol->get_data();
        const uint64_t * const inData = inDataCol->get_data();

        const size_t inDataSize = inDataCol->get_size_used_byte();
        // Exact allocation size (for uncompressed data).
        auto outGrCol = new column<uncompr_f>(inDataSize);
        // If no estimate is provided: Pessimistic allocation size (for
        // uncompressed data), reached only if there are as many groups as data
        // elements.
        auto outExtCol = new column<uncompr_f>(
                bool(outExtCountEstimate)
                ? (outExtCountEstimate * sizeof(uint64_t)) // use given estimate
                : inDataSize // use pessimistic estimate
        );
        uint64_t * outGr = outGrCol->get_data();
        uint64_t * outExt = outExtCol->get_data();
        const uint64_t * const initOutExt = outExt;

        // For both cases: Note that the []-operator creates a new entry with the
        // group-id(value) 0 in the map if the data item(key) is not found. Hence,
        // we use the group-id(value) 0 to indicate that the data item(key) was not
        // found. Consequently, we should not store a group-id(value) 0. Therefore,
        // the data items(keys) are mapped to the group-id(value) + 1 .
        if(inGrCol == nullptr) {
            // Unary group-operator.
            std::unordered_map<
                uint64_t,
                uint64_t
#ifndef MSV_NO_SELFMANAGED_MEMORY
                , std::hash<uint64_t>
                , std::equal_to<uint64_t>
                , global_scope_stdlib_allocator<std::pair<uint64_t, uint64_t >>
#endif
                > groupIds;
            for(unsigned i = 0; i < inDataCount; i++) {
                uint64_t & groupId = groupIds[inData[i]];
                if(!groupId) { // The data item(key) was not found.
                    groupId = groupIds.size();
                    *outExt = i;
                    outExt++;
                }
                *outGr = groupId - 1;
                outGr++;
            }
        }
        else {
            // Binary group-operator.
            // We have to use std::map, since std::pair is not hashable.
            std::map<std::pair<uint64_t, uint64_t>, uint64_t> groupIds;
            for(unsigned i = 0; i < inDataCount; i++) {
                uint64_t & groupId = groupIds[std::make_pair(inGr[i], inData[i])];
                if(!groupId) { // The data item(key) was not found.
                    groupId = groupIds.size();
                    *outExt = i;
                    outExt++;
                }
                *outGr = groupId - 1;
                outGr++;
            }
        }

        const size_t outExtCount = outExt - initOutExt;
        outGrCol->set_meta_data(inDataCount, inDataSize);
        outExtCol->set_meta_data(outExtCount, outExtCount * sizeof(uint64_t));

        return std::make_tuple(outGrCol, outExtCol);
    }
};

}
#endif //MORPHSTORE_CORE_OPERATORS_REFERENCE_GROUP_NEXT_H
