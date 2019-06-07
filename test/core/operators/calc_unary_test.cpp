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
 * @file calc_unary_test.cpp
 * @brief A little test/reference of the unary calculation-operator.
 * @todo TODOS?
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include "operator_test_frames.h"
#include <core/morphing/format.h>
#include <core/operators/scalar/calc_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <vector/scalar/extension_scalar.h>

using namespace morphstore;
using namespace vector;

/**
 * Unary functor for the "+5"-operation with an interface in the style of the
 * functors from the header \<functional\> in the standard library.
 */
template<class T>
struct plus_5 {
    T operator() (const T & p_val) const {
        return p_val + 5;
    }
};

int main(void) {
    const bool allGood = test_op_1in_1out(
            "Unary calculation",
            &calc_unary<
                    plus_5,
                    scalar<v64<uint64_t>>,
                    uncompr_f,
                    uncompr_f
            >
            make_column({10, 20, 0, 3, 100}),
            "inDataCol",
            make_column({15, 25, 5, 8, 105}),
            "outDataCol"
    );
    
    return !allGood;
}
