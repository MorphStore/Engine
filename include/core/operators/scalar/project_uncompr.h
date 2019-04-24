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
 * @file project_uncompr.h
 * @brief Template specialization of the project-operator for uncompressed
 * inputs and outputs using the scalar processing style. Note that these are
 * simple reference implementations not tailored for efficiency.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_PROJECT_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_PROJECT_UNCOMPR_H

#include <core/operators/interfaces/project.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/processing_style.h>

#include <cstdint>

namespace morphstore {
    
template<>
struct project_t<
        processing_style_t::scalar,
        uncompr_f,
        uncompr_f,
        uncompr_f
> {
    static
    const column<uncompr_f> *
    apply(
            const column<uncompr_f> * const inDataCol,
            const column<uncompr_f> * const inPosCol
    ) {
        const size_t inPosCount = inPosCol->get_count_values();
        const uint64_t * const inData = inDataCol->get_data();
        const uint64_t * const inPos = inPosCol->get_data();

        const size_t inPosSize = inPosCol->get_size_used_byte();
        // Exact allocation size (for uncompressed data).
        auto outDataCol = new column<uncompr_f>(inPosSize);
        uint64_t * outData = outDataCol->get_data();

        for(unsigned i = 0; i < inPosCount; i++) {
            *outData = inData[inPos[i]];
            outData++;
        }

        outDataCol->set_meta_data(inPosCount, inPosSize);

        return outDataCol;
    }
};

}
#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_PROJECT_UNCOMPR_H