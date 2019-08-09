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
 * @file vbp_test.cpp
 * @brief Tests of the (de)compression morph operators for the formats related
 * to vertical bit-packing, i.e., `static_vbp_f` and `dynamic_vbp_f`.
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/static_vbp.h>
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


// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(vector_extension, format) { \
    new typename t_varex_t::operator_wrapper::template for_output_formats<format>::template for_input_formats<format>( \
        &morph<vector_extension, format, format>, true \
    ), \
    STR_EVAL_MACROS(vector_extension), \
    STR_EVAL_MACROS(format), \
}

template<class t_varex_t, unsigned t_Bw>
std::vector<typename t_varex_t::variant_t> makeVariants() {
    return {
        // Uncompressed reference variant, required to check the correctness of
        // the compressed variants.
        MAKE_VARIANT(scalar<v64<uint64_t>>, uncompr_f),
                
        // Compressed variants.
#ifdef AVX512
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(static_vbp_f<t_Bw, 8>)),
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<512, 64, 8>)),
#endif
#ifdef AVXTWO
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(static_vbp_f<t_Bw, 4>)),
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<256, 32, 4>)),
#endif
#ifdef SSE
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(static_vbp_f<t_Bw, 2>)),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<128, 16, 2>)),
#endif
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<t_Bw, 1>)),
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<64, 8, 1>))
    };
}

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 1>::type
        ::for_variant_params<std::string, std::string>
        ::for_setting_params<unsigned, size_t>;
    varex_t varex(
            {},
            {"vector_extension", "format"},
            {"bitwidth", "countValues"}
    );
    
    for(unsigned bw = 1; bw <= std::numeric_limits<uint64_t>::digits; bw++) {
        std::vector<varex_t::variant_t> variants;
        switch(bw) {
            // Generated with python:
            // for bw in range(1, 64+1):
            //   print("            case {: >2}: variants = makeVariants<varex_t, {: >2}>(); break;".format(bw, bw))
            case  1: variants = makeVariants<varex_t,  1>(); break;
            case  2: variants = makeVariants<varex_t,  2>(); break;
            case  3: variants = makeVariants<varex_t,  3>(); break;
            case  4: variants = makeVariants<varex_t,  4>(); break;
            case  5: variants = makeVariants<varex_t,  5>(); break;
            case  6: variants = makeVariants<varex_t,  6>(); break;
            case  7: variants = makeVariants<varex_t,  7>(); break;
            case  8: variants = makeVariants<varex_t,  8>(); break;
            case  9: variants = makeVariants<varex_t,  9>(); break;
            case 10: variants = makeVariants<varex_t, 10>(); break;
            case 11: variants = makeVariants<varex_t, 11>(); break;
            case 12: variants = makeVariants<varex_t, 12>(); break;
            case 13: variants = makeVariants<varex_t, 13>(); break;
            case 14: variants = makeVariants<varex_t, 14>(); break;
            case 15: variants = makeVariants<varex_t, 15>(); break;
            case 16: variants = makeVariants<varex_t, 16>(); break;
            case 17: variants = makeVariants<varex_t, 17>(); break;
            case 18: variants = makeVariants<varex_t, 18>(); break;
            case 19: variants = makeVariants<varex_t, 19>(); break;
            case 20: variants = makeVariants<varex_t, 20>(); break;
            case 21: variants = makeVariants<varex_t, 21>(); break;
            case 22: variants = makeVariants<varex_t, 22>(); break;
            case 23: variants = makeVariants<varex_t, 23>(); break;
            case 24: variants = makeVariants<varex_t, 24>(); break;
            case 25: variants = makeVariants<varex_t, 25>(); break;
            case 26: variants = makeVariants<varex_t, 26>(); break;
            case 27: variants = makeVariants<varex_t, 27>(); break;
            case 28: variants = makeVariants<varex_t, 28>(); break;
            case 29: variants = makeVariants<varex_t, 29>(); break;
            case 30: variants = makeVariants<varex_t, 30>(); break;
            case 31: variants = makeVariants<varex_t, 31>(); break;
            case 32: variants = makeVariants<varex_t, 32>(); break;
            case 33: variants = makeVariants<varex_t, 33>(); break;
            case 34: variants = makeVariants<varex_t, 34>(); break;
            case 35: variants = makeVariants<varex_t, 35>(); break;
            case 36: variants = makeVariants<varex_t, 36>(); break;
            case 37: variants = makeVariants<varex_t, 37>(); break;
            case 38: variants = makeVariants<varex_t, 38>(); break;
            case 39: variants = makeVariants<varex_t, 39>(); break;
            case 40: variants = makeVariants<varex_t, 40>(); break;
            case 41: variants = makeVariants<varex_t, 41>(); break;
            case 42: variants = makeVariants<varex_t, 42>(); break;
            case 43: variants = makeVariants<varex_t, 43>(); break;
            case 44: variants = makeVariants<varex_t, 44>(); break;
            case 45: variants = makeVariants<varex_t, 45>(); break;
            case 46: variants = makeVariants<varex_t, 46>(); break;
            case 47: variants = makeVariants<varex_t, 47>(); break;
            case 48: variants = makeVariants<varex_t, 48>(); break;
            case 49: variants = makeVariants<varex_t, 49>(); break;
            case 50: variants = makeVariants<varex_t, 50>(); break;
            case 51: variants = makeVariants<varex_t, 51>(); break;
            case 52: variants = makeVariants<varex_t, 52>(); break;
            case 53: variants = makeVariants<varex_t, 53>(); break;
            case 54: variants = makeVariants<varex_t, 54>(); break;
            case 55: variants = makeVariants<varex_t, 55>(); break;
            case 56: variants = makeVariants<varex_t, 56>(); break;
            case 57: variants = makeVariants<varex_t, 57>(); break;
            case 58: variants = makeVariants<varex_t, 58>(); break;
            case 59: variants = makeVariants<varex_t, 59>(); break;
            case 60: variants = makeVariants<varex_t, 60>(); break;
            case 61: variants = makeVariants<varex_t, 61>(); break;
            case 62: variants = makeVariants<varex_t, 62>(); break;
            case 63: variants = makeVariants<varex_t, 63>(); break;
            case 64: variants = makeVariants<varex_t, 64>(); break;
        }

        for(size_t countValues : {
            // The following numbers of data elements represent all
            // combinations of the following features for all vector
            // extensions:
            // - uncompressed rest (17)
            // - complete pages of dynamic_vbp_f
            //   (vector size [byte] * vector size [bit])
            // - incomplete pages of dynamic_vbp_f (3 * vector size [bit])
            17,
            // for scalar
            3 * 64,
            3 * 64 + 17,
            10 * 8 * 64,
            10 * 8 * 64 + 3 * 64,
            10 * 8 * 64 + 17,
            10 * 8 * 64 + 3 * 64 + 17,
            // for sse
            3 * 128,
            3 * 128 + 17,
            10 * 16 * 128,
            10 * 16 * 128 + 3 * 128,
            10 * 16 * 128 + 3 * 128 + 17,
            // for avx2
            3 * 256,
            3 * 256 + 17,
            10 * 32 * 256,
            10 * 32 * 256 + 3 * 256,
            10 * 32 * 256 + 3 * 256 + 17
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