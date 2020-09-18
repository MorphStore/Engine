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
 * @file group_first.h
 * @brief Template specialization of the group_first-operator for uncompressed
 * inputs and outputs using the scalar processing style. Note that this is a
 * simple reference implementations not tailored for efficiency.
 * 
 * Note that this reference implementation of the group_first-operator
 * delegates to the reference implementation of the group_next-operator.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_REFERENCE_GROUP_FIRST_H
#define MORPHSTORE_CORE_OPERATORS_REFERENCE_GROUP_FIRST_H

#include <core/operators/interfaces/group_first.h>
#include <core/operators/reference/group_next.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#ifndef MSV_NO_SELFMANAGED_MEMORY
#include <core/memory/management/allocators/global_scope_allocator.h>
#endif

#include <cstdint>
#include <tuple>

namespace morphstore {

template<>
struct group_first_t<
        vectorlib::scalar<vectorlib::v64<uint64_t>>,
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
            const column<uncompr_f> * const inDataCol,
            const size_t outExtCountEstimate
    ) {
        return group_next<
                vectorlib::scalar<vectorlib::v64<uint64_t>>,
                uncompr_f,
                uncompr_f,
                uncompr_f
        >(
                nullptr,
                inDataCol,
                outExtCountEstimate
        );
    }
};

}
#endif //MORPHSTORE_CORE_OPERATORS_REFERENCE_GROUP_FIRST_H
