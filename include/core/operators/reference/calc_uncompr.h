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
 * @file calc_uncompr.h
 * @brief Template specializations of the unary and binary
 * calculation-operators for uncompressed inputs and outputs using the scalar
 * processing style. Note that these are simple reference implementations not
 * tailored for efficiency.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_CALC_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_CALC_UNCOMPR_H

#include <core/operators/interfaces/calc.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#include <cstdint>
#include <stdexcept>

namespace morphstore {
    
template<template<typename> class t_unary_op>
struct calc_unary_t<
        t_unary_op,
        vectorlib::scalar<vectorlib::v64<uint64_t>>,
        uncompr_f,
        uncompr_f
> {
    static
    const column<uncompr_f> * apply(
            const column<uncompr_f> * const inDataCol
    ) {
        const size_t inDataCount = inDataCol->get_count_values();
        const uint64_t * const inData = inDataCol->get_data();
        
        const size_t inDataSize = inDataCol->get_size_used_byte();
        // Exact allocation size (for uncompressed data).
        auto outDataCol = new column<uncompr_f>(inDataSize);
        uint64_t * const outData = outDataCol->get_data();
        
        t_unary_op<uint64_t> op;
        for(unsigned i = 0; i < inDataCount; i++)
            outData[i] = op(inData[i]);
        
        outDataCol->set_meta_data(inDataCount, inDataSize);
        
        return outDataCol;
    }
};
    
template<template<typename> class t_binary_op>
struct calc_binary_t<
        t_binary_op, vectorlib::scalar<vectorlib::v64<uint64_t>>,
        uncompr_f,
        uncompr_f,
        uncompr_f
>  {
    static
    const column<uncompr_f> * apply(
            const column<uncompr_f> * const inDataLCol,
            const column<uncompr_f> * const inDataRCol
    ) {
        const size_t inDataCount = inDataLCol->get_count_values();
    
        if(inDataCount != inDataRCol->get_count_values())
            throw std::runtime_error(
                    "calc: inDataLCol and inDataRCol must contain the same "
                    "number of data elements"
            );
        
        const uint64_t * const inDataL = inDataLCol->get_data();
        const uint64_t * const inDataR = inDataRCol->get_data();
        
        const size_t inDataSize = inDataLCol->get_size_used_byte();
        // Exact allocation size (for uncompressed data).
        auto outDataCol = new column<uncompr_f>(inDataSize);
        uint64_t * const outData = outDataCol->get_data();
        
        t_binary_op<uint64_t> op;
        for(unsigned i = 0; i < inDataCount; i++)
            outData[i] = op(inDataL[i], inDataR[i]);
        
        outDataCol->set_meta_data(inDataCount, inDataSize);
        
        return outDataCol;
    }
};

}
#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_CALC_UNCOMPR_H
