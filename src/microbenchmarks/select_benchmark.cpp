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
 * @file select_benchmark.cpp
 * @brief A little mirco benchmark of the select operator on uncompressed and
 * compressed data.
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/uncompr.h>
#include <core/operators/general_vectorized/select_compr.h>
#include <core/operators/scalar/select_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/variant_executor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <functional>
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

#define STATIC_VBP_NAME "static_vbp_f<>"
#define STATIC_VBP_FORMAT(vector_extension, bw) \
    SINGLE_ARG(static_vbp_f<bw, vector_extension::vector_helper_t::element_count::value>)

#define DYNAMIC_VBP_NAME "dynamic_vbp_f<>"
#define DYNAMIC_VBP_FORMAT(vector_extension) \
    SINGLE_ARG(dynamic_vbp_f< \
            vector_extension::vector_helper_t::size_bit::value, \
            vector_extension::vector_helper_t::size_byte::value, \
            vector_extension::vector_helper_t::element_count::value \
    >)

#define MAKE_VARIANT_CLASSICAL(vector_extension, out_data_f, outDataFName, in_data_f, inDataFName, bw) { \
    new typename t_varex_t::operator_wrapper::template for_output_formats<out_data_f>::template for_input_formats<in_data_f>( \
        &morphstore::select<std::equal_to, vector_extension, out_data_f, in_data_f> \
    ), \
    "classical", \
    STR_EVAL_MACROS(vector_extension), \
    STR_EVAL_MACROS(outDataFName), \
    STR_EVAL_MACROS(inDataFName), \
    bw \
}

#define MAKE_VARIANT_STRUCT_WIT(vector_extension, out_pos_f, outPosFName, in_data_f, inDataFName, bw) { \
    new typename t_varex_t::operator_wrapper::template for_output_formats<out_pos_f>::template for_input_formats<in_data_f>( \
        &my_select_wit_t<equal, vector_extension, out_pos_f, in_data_f>::apply \
    ), \
    "my_select_wit", \
    STR_EVAL_MACROS(vector_extension), \
    STR_EVAL_MACROS(outPosFName), \
    STR_EVAL_MACROS(inDataFName), \
    bw \
}

#define MAKE_VARIANTS_WIT(vector_extension, bw) \
    MAKE_VARIANT_STRUCT_WIT( \
            vector_extension, \
            uncompr_f, "uncompr_f", \
            uncompr_f, "uncompr_f", \
            bw \
    ), \
    MAKE_VARIANT_STRUCT_WIT( \
            vector_extension, \
            uncompr_f, "uncompr_f", \
            STATIC_VBP_FORMAT(vector_extension, bw), STATIC_VBP_NAME, \
            bw \
    ), \
    MAKE_VARIANT_STRUCT_WIT( \
            vector_extension, \
            STATIC_VBP_FORMAT(vector_extension, bw), STATIC_VBP_NAME, \
            uncompr_f, "uncompr_f", \
            bw \
    ), \
    MAKE_VARIANT_STRUCT_WIT( \
            vector_extension, \
            STATIC_VBP_FORMAT(vector_extension, bw), STATIC_VBP_NAME, \
            STATIC_VBP_FORMAT(vector_extension, bw), STATIC_VBP_NAME, \
            bw \
    ), \
    MAKE_VARIANT_STRUCT_WIT( \
            vector_extension, \
            uncompr_f, "uncompr_f", \
            DYNAMIC_VBP_FORMAT(vector_extension), DYNAMIC_VBP_NAME, \
            bw \
    ), \
    MAKE_VARIANT_STRUCT_WIT( \
            vector_extension, \
            DYNAMIC_VBP_FORMAT(vector_extension), DYNAMIC_VBP_NAME, \
            uncompr_f, "uncompr_f", \
            bw \
    ), \
    MAKE_VARIANT_STRUCT_WIT( \
            vector_extension, \
            DYNAMIC_VBP_FORMAT(vector_extension), DYNAMIC_VBP_NAME, \
            DYNAMIC_VBP_FORMAT(vector_extension), DYNAMIC_VBP_NAME, \
            bw \
    )

template<class t_varex_t, unsigned t_Bw>
std::vector<typename t_varex_t::variant_t> make_variants() {
    return {
        // Uncompressed reference variant, required to check the correctness of
        // the compressed variants.
        MAKE_VARIANT_CLASSICAL(scalar<v64<uint64_t>>, uncompr_f, "uncompr_f", uncompr_f, "uncompr_f", t_Bw),
        
        // Compressed variants.
#ifdef AVX512
        MAKE_VARIANTS_WIT(avx512<v512<uint64_t>>, t_Bw),
#endif
#ifdef AVXTWO
        MAKE_VARIANTS_WIT(avx2<v256<uint64_t>>, t_Bw),
#endif
#ifdef SSE
        MAKE_VARIANTS_WIT(sse<v128<uint64_t>>, t_Bw),
#endif
        MAKE_VARIANTS_WIT(scalar<v64<uint64_t>>, t_Bw)
    };
}

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 1, uint64_t, size_t>::type
        ::for_variant_params<std::string, std::string, std::string, std::string, unsigned>
        ::for_setting_params<size_t, float>;
    varex_t varex(
            {"pred", "est"},
            {"variant", "vector_extension", "out_pos_f", "in_data_f", "bw"},
            {"countValues", "selectivity"}
    );
    
    const uint64_t pred = 0;
    
    for(float selectivity : {
        0.1,
        0.5,
        0.9
    })
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
            for(
                    // We start at the bit width required for the greatest
                    // position.
                    unsigned bw = effective_bitwidth(countValues - 1);
                    bw <= std::numeric_limits<uint64_t>::digits;
                    bw++
            ) {
                std::vector<varex_t::variant_t> variants;
                switch(bw) {
                    // Generated with python:
                    // for bw in range(1, 64+1):
                    //   print("case {: >2}: variants = make_variants<varex_t, {: >2}>(); break;".format(bw, bw))
                    case  1: variants = make_variants<varex_t,  1>(); break;
                    case  2: variants = make_variants<varex_t,  2>(); break;
                    case  3: variants = make_variants<varex_t,  3>(); break;
                    case  4: variants = make_variants<varex_t,  4>(); break;
                    case  5: variants = make_variants<varex_t,  5>(); break;
                    case  6: variants = make_variants<varex_t,  6>(); break;
                    case  7: variants = make_variants<varex_t,  7>(); break;
                    case  8: variants = make_variants<varex_t,  8>(); break;
                    case  9: variants = make_variants<varex_t,  9>(); break;
                    case 10: variants = make_variants<varex_t, 10>(); break;
                    case 11: variants = make_variants<varex_t, 11>(); break;
                    case 12: variants = make_variants<varex_t, 12>(); break;
                    case 13: variants = make_variants<varex_t, 13>(); break;
                    case 14: variants = make_variants<varex_t, 14>(); break;
                    case 15: variants = make_variants<varex_t, 15>(); break;
                    case 16: variants = make_variants<varex_t, 16>(); break;
                    case 17: variants = make_variants<varex_t, 17>(); break;
                    case 18: variants = make_variants<varex_t, 18>(); break;
                    case 19: variants = make_variants<varex_t, 19>(); break;
                    case 20: variants = make_variants<varex_t, 20>(); break;
                    case 21: variants = make_variants<varex_t, 21>(); break;
                    case 22: variants = make_variants<varex_t, 22>(); break;
                    case 23: variants = make_variants<varex_t, 23>(); break;
                    case 24: variants = make_variants<varex_t, 24>(); break;
                    case 25: variants = make_variants<varex_t, 25>(); break;
                    case 26: variants = make_variants<varex_t, 26>(); break;
                    case 27: variants = make_variants<varex_t, 27>(); break;
                    case 28: variants = make_variants<varex_t, 28>(); break;
                    case 29: variants = make_variants<varex_t, 29>(); break;
                    case 30: variants = make_variants<varex_t, 30>(); break;
                    case 31: variants = make_variants<varex_t, 31>(); break;
                    case 32: variants = make_variants<varex_t, 32>(); break;
                    case 33: variants = make_variants<varex_t, 33>(); break;
                    case 34: variants = make_variants<varex_t, 34>(); break;
                    case 35: variants = make_variants<varex_t, 35>(); break;
                    case 36: variants = make_variants<varex_t, 36>(); break;
                    case 37: variants = make_variants<varex_t, 37>(); break;
                    case 38: variants = make_variants<varex_t, 38>(); break;
                    case 39: variants = make_variants<varex_t, 39>(); break;
                    case 40: variants = make_variants<varex_t, 40>(); break;
                    case 41: variants = make_variants<varex_t, 41>(); break;
                    case 42: variants = make_variants<varex_t, 42>(); break;
                    case 43: variants = make_variants<varex_t, 43>(); break;
                    case 44: variants = make_variants<varex_t, 44>(); break;
                    case 45: variants = make_variants<varex_t, 45>(); break;
                    case 46: variants = make_variants<varex_t, 46>(); break;
                    case 47: variants = make_variants<varex_t, 47>(); break;
                    case 48: variants = make_variants<varex_t, 48>(); break;
                    case 49: variants = make_variants<varex_t, 49>(); break;
                    case 50: variants = make_variants<varex_t, 50>(); break;
                    case 51: variants = make_variants<varex_t, 51>(); break;
                    case 52: variants = make_variants<varex_t, 52>(); break;
                    case 53: variants = make_variants<varex_t, 53>(); break;
                    case 54: variants = make_variants<varex_t, 54>(); break;
                    case 55: variants = make_variants<varex_t, 55>(); break;
                    case 56: variants = make_variants<varex_t, 56>(); break;
                    case 57: variants = make_variants<varex_t, 57>(); break;
                    case 58: variants = make_variants<varex_t, 58>(); break;
                    case 59: variants = make_variants<varex_t, 59>(); break;
                    case 60: variants = make_variants<varex_t, 60>(); break;
                    case 61: variants = make_variants<varex_t, 61>(); break;
                    case 62: variants = make_variants<varex_t, 62>(); break;
                    case 63: variants = make_variants<varex_t, 63>(); break;
                    case 64: variants = make_variants<varex_t, 64>(); break;
                }
                
                varex.print_datagen_started();
                const size_t countMatches = static_cast<size_t>(
                        static_cast<float>(countValues) * selectivity
                );
                auto origCol = generate_exact_number(
                        countValues,
                        countMatches,
                        pred,
                        bitwidth_max<uint64_t>(bw)
                );
                varex.print_datagen_done();

                varex.execute_variants(
                        variants, countValues, selectivity, origCol, pred, 0
                );

                delete origCol;
            }
        }
    
    varex.done();
    
    return 0;
}