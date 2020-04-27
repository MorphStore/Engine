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
 * @file select_sum_query.cpp
 * @brief A little example query with a selection on one column and a sum on
 * another column.
 */
#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/operators/general_vectorized/agg_sum_compr.h>
#include <core/operators/general_vectorized/project_compr.h>
#include <core/operators/general_vectorized/select_compr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>
#include <core/utils/math.h>

#include <core/morphing/k_wise_ns.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/vbp.h>
#include <core/morphing/morph.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <functional>
#include <iostream>
#include <random>

#include <core/morphing/type_packing.h>


#include <core/operators/general_vectorized/agg_sum_uncompr.h>
#include <core/operators/general_vectorized/project_uncompr.h>
#include <core/operators/general_vectorized/select_uncompr.h>

#include "../../include/core/storage/column.h"
#include "../../include/core/storage/column_gen.h"
#include "../../include/core/utils/basic_types.h"
#include "../../include/core/utils/printing.h"

using namespace morphstore;
using namespace vectorlib;

// ****************************************************************************
// * Example query
// ****************************************************************************


int main( void ) {
    // ************************************************************************
    // * Generation of the synthetic base data
    // ************************************************************************
    
    std::cout << "Base data generation started... ";
    std::cout.flush();
    
    const size_t countValues = 10; 
    const column<uncompr_f> * const baseCol1 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(100, 199),
            false,
            2
    );

    print_columns(print_buffer_base::hexadecimal, baseCol1, "SUM(baseCol1)");
  //  using ve = scalar<v64<uint64_t> >;
    using ve = sse<v128<uint64_t> >;

    //compression
    auto baseCol1Compr = morph<ve, type_packing_f<uint32_t > >(baseCol1);
    std::cout << "get_size_compr_byte baseCol1 " << baseCol1Compr->get_size_compr_byte() << std::endl;
    print_columns(print_buffer_base::hexadecimal, baseCol1Compr, "SUM(baseCol1)");
    //decompression
    auto baseCol1Decompr = morph<ve, uncompr_f >(baseCol1Compr);
    print_columns(print_buffer_base::hexadecimal, baseCol1Decompr, "baseCol1Decompr)");

    std::cout << "done." << std::endl;
    

    return 0;
}




