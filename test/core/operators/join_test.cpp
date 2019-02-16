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
 * @file join_test.cpp
 * @brief A little test/reference of the join-operator.
 * @todo TODOS?
 */

// This must be included first to allow compilation.
#include "../../../include/core/memory/mm_glob.h"

#include "operator_test_frames.h"
#include "../../../include/core/morphing/format.h"
#include "../../../include/core/operators/scalar/join_uncompr.h"
#include "../../../include/core/storage/column.h"
#include "../../../include/core/storage/column_gen.h"
#include "../../../include/core/utils/processing_style.h"

using namespace morphstore;

int main( void ) {
    test_op_2in_2out_1val(
            "Join",
            &nested_loop_join<
                    processing_style_t::scalar,
                    uncompr_f,
                    uncompr_f
            >,
            make_column({22, 44, 11, 22, 55, 77}),
            make_column({33, 22, 22, 11}),
            "inDataLCol",
            "inDataRCol",
            make_column({0, 0, 2, 3, 3}),
            make_column({1, 2, 3, 1, 2}),
            "outPosLCol",
            "outPosRCol",
            0 // use pessimistic output size estimation
    );
            
    return 0;
}