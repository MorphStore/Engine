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
 * @file project_test.cpp
 * @brief A little test/reference of the project-operator.
 * @todo TODOS?
 */

// This must be included first to allow compilation.
#include "../../../include/core/memory/mm_glob.h"

#include "operator_test_frames.h"
#include "../../../include/core/morphing/format.h"
#include "../../../include/core/operators/scalar/project_uncompr.h"
#include "../../../include/core/storage/column.h"
#include "../../../include/core/storage/column_gen.h"
#include "../../../include/core/utils/processing_style.h"

using namespace morphstore;

int main( void ) {
    test_op_2in_1out(
            "Project",
            &project<processing_style_t::scalar, uncompr_f>,
            make_column({11, 44, 22, 33, 11}),
            make_column({1, 3, 4}),
            "inDataCol",
            "inPosCol",
            make_column({44, 33, 11}),
            "outPosCol"
    );
    
    return 0;
}