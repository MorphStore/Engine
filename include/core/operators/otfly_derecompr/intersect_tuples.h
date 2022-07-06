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
 * @file intersect_tuple.h
 * @brief Intersect operator for tuples to enable combining tuples of position lists of columns for the natural join.
 * @todo TODOS?
 */

#ifndef DAPHNE_PROTOTYPE_INTERSECT_TUPLE_H
#define DAPHNE_PROTOTYPE_INTERSECT_TUPLE_H

#include <core/operators/interfaces/intersect_tuples.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <unordered_map>


namespace morphstore{

    template<>
    struct intersect_tuples_t<uncompr_f, uncompr_f> {
    static
    const std::tuple<
            const column<uncompr_f> *,
            const column<uncompr_f> *
    >
    apply(const std::tuple<const column<uncompr_f> *,
            const column<uncompr_f> *> posTupleListLeft, const std::tuple<const column<uncompr_f> *,
            const column<uncompr_f> *> posTupleListRight, const size_t outPosCountEstimate) {
        const column<uncompr_f> *posColLeftLeft= nullptr;
        const column<uncompr_f> *posColLeftRight= nullptr;
        const column<uncompr_f> *posColRightLeft= nullptr;
        const column<uncompr_f> *posColRightRight= nullptr;
        posColLeftLeft = std::get<0>(posTupleListLeft);
        posColLeftRight = std::get<1>(posTupleListLeft);
        posColRightLeft = std::get<0>(posTupleListRight);
        posColRightRight = std::get<1>(posTupleListRight);

        const uint64_t * inPosLL = posColLeftLeft->get_data();
        const uint64_t * inPosLR = posColLeftRight->get_data();
        const uint64_t * inPosRL = posColRightLeft->get_data();
        const uint64_t * inPosRR = posColRightRight->get_data();
        const uint64_t * const endInPosLL = inPosLL + posColLeftLeft->get_count_values();
        const uint64_t * const endInPosRL = inPosRL + posColRightLeft->get_count_values();

        // column are contained in the larger input column as well.
        auto outPosColLeft = new column<uncompr_f>(
                bool(outPosCountEstimate)
                // use given estimate
                ? (outPosCountEstimate * sizeof(uint64_t))
                // use pessimistic estimate
                : std::min(
                        posColLeftLeft->get_size_used_byte(),
                        posColRightLeft->get_size_used_byte()
                )
        );
        uint64_t * outPosLeft = outPosColLeft->get_data();
        const uint64_t * const initOutPosLeft = outPosLeft;
        // column are contained in the larger input column as well.
        auto outPosColRight = new column<uncompr_f>(
                bool(outPosCountEstimate)
                // use given estimate
                ? (outPosCountEstimate * sizeof(uint64_t))
                // use pessimistic estimate
                : std::min(
                        posColLeftLeft->get_size_used_byte(),
                        posColRightRight->get_size_used_byte()
                )
        );
        uint64_t * outPosRight = outPosColRight->get_data();

        while(inPosLL < endInPosLL && inPosRL < endInPosRL) {
            if (*inPosLR < *inPosRR) {
                inPosLL++;
                inPosLR++;
            } else if (*inPosRR < *inPosLR) {
                inPosRL++;
                inPosRR++;
            } else if (*inPosLL != *inPosRL) {
                // right side is unique, so if left side is not the same we have no possible matches
                inPosLL++;
                inPosRL++;
                inPosLR++;
                inPosRR++;
            } else { // *inPosLR == *inPosRR && *inPosLL == *inPosRL
                *outPosLeft = *inPosLL;
                outPosLeft++;
                *outPosRight = *inPosLR;
                outPosRight++;
                inPosLL++;
                inPosRL++;
                inPosLR++;
                inPosRR++;
            }
        }

        const size_t outPosCount = outPosLeft - initOutPosLeft;
        outPosColLeft->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
        outPosColRight->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));

        return std::make_tuple(outPosColLeft, outPosColRight);
    }

};
}

#endif //DAPHNE_PROTOTYPE_INTERSECT_TUPLE_H
