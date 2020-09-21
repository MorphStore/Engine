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
 * @file select.h
 * @brief Template specialization of the select-operator for uncompressed
 * inputs and outputs using the scalar processing style. Note that these are
 * simple reference implementations not tailored for efficiency.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_SELECT_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_SELECT_H

#include <core/operators/interfaces/select.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/scalar/primitives/compare_scalar.h>

#include <cstdint>

namespace morphstore {
    
template<template< class, int > class t_op>
struct select_t<vectorlib::scalar<vectorlib::v64<uint64_t>>, t_op, uncompr_f, uncompr_f> {
    using VectorExtension =  vectorlib::scalar<vectorlib::v64<uint64_t>>;
    static
    const column<uncompr_f> * apply(
            const column<uncompr_f> * const inDataCol,
            const uint64_t val,
            const size_t outPosCountEstimate = 0
    ) {
        const size_t inDataCount = inDataCol->get_count_values();
        const uint64_t * const inData = inDataCol->get_data();

        // If no estimate is provided: Pessimistic allocation size (for
        // uncompressed data), reached only if all input data elements pass the
        // selection.
        auto outPosCol = new column<uncompr_f>(
                bool(outPosCountEstimate)
                // use given estimate
                ? (outPosCountEstimate * sizeof(uint64_t))
                // use pessimistic estimate
                : inDataCol->get_size_used_byte()
        );
        
      //  t_op<uint64_t> op;
        uint64_t * outPos = outPosCol->get_data();
        const uint64_t * const initOutPos = outPos;


        for(unsigned i = 0; i < inDataCount; i++)
          if (t_op<VectorExtension,VectorExtension::vector_helper_t::granularity::value>::apply(inData[i], val) ){
        //    if(op(inData[i], val)) {
                *outPos = i;
                outPos++;
            }

        const size_t outPosCount = outPos - initOutPos;
        outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));

        return outPosCol;
    }
};

}
#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_SELECT_H
