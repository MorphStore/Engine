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
 * @file delta_test.cpp
 * @brief Tests of the (de)compression morph operators for `delta_f`, the
 * cascade of delta coding with any format.
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/delta.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/vbp.h>
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


// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(vector_extension, format, formatName) { \
    new typename varex_t::operator_wrapper::template for_output_formats<format>::template for_input_formats<format>( \
        &morph<vector_extension, format, format>, true \
    ), \
    STR_EVAL_MACROS(vector_extension), \
    STR_EVAL_MACROS(formatName), \
}

// Packing the deltas with 64 bits is, of course, not useful at all. However,
// this is just for testing if the cascade works.
#define MAKE_VARIANT_DELTA_VBP(vector_extension) \
    MAKE_VARIANT( \
            vector_extension, \
            SINGLE_ARG(delta_f< \
                    1024, \
                    vector_extension::vector_helper_t::element_count::value, \
                    vbp_l< \
                            64, \
                            vector_extension::vector_helper_t::element_count::value \
                    > \
            >), \
            SINGLE_ARG(delta_f<1024, step, vbp_l<64, step> >) \
    )

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 1>::type
        ::for_variant_params<std::string, std::string>
        ::for_setting_params<unsigned, size_t, bool>;
    varex_t varex(
            {},
            {"vector_extension", "format"},
            {"bitwidth", "countValues", "isSorted"}
    );
    
    std::vector<varex_t::variant_t> variants = {
        // Uncompressed reference.
        MAKE_VARIANT(scalar<v64<uint64_t>>, uncompr_f, uncompr_f),
        
        // Compressed variants.
        MAKE_VARIANT_DELTA_VBP(scalar<v64<uint64_t>>),
#ifdef SSE
        MAKE_VARIANT_DELTA_VBP(sse<v128<uint64_t>>),
#endif
#ifdef AVXTWO
        MAKE_VARIANT_DELTA_VBP(avx2<v256<uint64_t>>),
#endif
#ifdef AVX512
        MAKE_VARIANT_DELTA_VBP(avx512<v512<uint64_t>>)
#endif
    };
    
    for(unsigned bw = 1; bw <= std::numeric_limits<uint64_t>::digits; bw++) {
        for(size_t countValues : {
            123, 1024, 1234, 10240
        }) {
            for(bool isSorted : {false, true}) {
                varex.print_datagen_started();
                auto origCol = generate_with_distr(
                        countValues,
                        std::uniform_int_distribution<uint64_t>(
                                0, bitwidth_max<uint64_t>(bw)
                        ),
                        isSorted
                );
                varex.print_datagen_done();

                varex.execute_variants(
                        variants, bw, countValues, isSorted, origCol
                );

//                delete origCol;
            }
        }
    }
    
    varex.done();
    
    return !varex.good();
}