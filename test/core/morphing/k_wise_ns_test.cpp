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
 * @file k_wise_ns_test.cpp
 * @brief Tests of the (de)compression morph operators for k-Wise Null
 * Suppression, i.e., for the format `k_wise_ns_f`.
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/k_wise_ns.h>
#include <core/morphing/uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/variant_executor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vectorlib;

#ifdef SSE

// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(format) { \
    new typename varex_t::operator_wrapper::template for_output_formats<format>::template for_input_formats<format>( \
        &morph<sse<v128<uint64_t> >, format, format>, true \
    ), \
    STR_EVAL_MACROS(format), \
}

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 1>::type
        ::for_variant_params<std::string>
        ::for_setting_params<unsigned, size_t>;
    varex_t varex(
            {},
            {"format"},
            {"bitwidth", "countValues"}
    );
    
    std::vector<varex_t::variant_t> variants = {
        MAKE_VARIANT(uncompr_f),
        // 2 == sse<v128<uint64_t>>::vector_helper_t::element_count::value
        // but the long expression would look ugly in the output...
        MAKE_VARIANT(k_wise_ns_f<2>)
    };
    
    for(unsigned bw = 1; bw <= std::numeric_limits<uint64_t>::digits; bw++) {
        for(size_t countValues : {
            1, 2, 100, 101
        }) {
            varex.print_datagen_started();
            auto origCol = generate_with_distr(
                    countValues,
                    std::uniform_int_distribution<uint64_t>(
                            0, bitwidth_max<uint64_t>(bw)
                    ),
                    false
            );
            varex.print_datagen_done();

            varex.execute_variants(variants, bw, countValues, origCol);

//            delete origCol;
        }
    }
    
    varex.done();
    
    return !varex.good();
}
#else
int main() {
    return 0;
}
#endif