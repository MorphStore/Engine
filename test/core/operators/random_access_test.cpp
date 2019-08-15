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
 * @file random_access_test.cpp
 * @brief Tests of the implementations of the random_read_access-interface.
 */

#include <core/memory/noselfmanaging_helper.h>
#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/vbp.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/variant_executor.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include <cstdint>
#include <cstring>

using namespace morphstore;
using namespace vectorlib;


// ****************************************************************************
// Some simple operatos for testing the random access interface.
// ****************************************************************************

// As the reference.
const column<uncompr_f> * copy(
        const column<uncompr_f> * p_InDataCol,
        const column<uncompr_f> * p_InPosCol
) {
    const uint8_t * inData = p_InDataCol->get_data();
    
    const size_t inPosCount = p_InPosCol->get_count_values();
    
    const size_t outSize = get_size_max_byte_any_len<uncompr_f>(inPosCount);
    auto outDataCol = new column<uncompr_f>(outSize);
    uint8_t * outData = outDataCol->get_data();
    
    memcpy(outData, inData, outSize);
    
    outDataCol->set_meta_data(inPosCount, outSize);
    return outDataCol;
}

// Simplified project-operator.
template<class t_ve, class t_in_data_f>
const column<uncompr_f> * simple_project(
        const column<t_in_data_f> * p_InDataCol,
        const column<uncompr_f> * p_InPosCol
) {
    IMPORT_VECTOR_BOILER_PLATE(t_ve)
    
    random_read_access<t_ve, t_in_data_f> rra(p_InDataCol->get_data());
    
    const base_t * inPos = p_InPosCol->get_data();
    const size_t inPosCount = p_InPosCol->get_count_values();
    
    const size_t outSize = get_size_max_byte_any_len<uncompr_f>(inPosCount);
    auto outDataCol = new column<uncompr_f>(outSize);
    base_t * outData = outDataCol->get_data();
    
    for(
            size_t inPosIdx = 0;
            inPosIdx < inPosCount;
            inPosIdx += vector_element_count::value
    ) {
        store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                outData,
                rra.get(load<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        inPos + inPosIdx
                ))
        );
        outData += vector_element_count::value;
    }
    
    outDataCol->set_meta_data(inPosCount, outSize);
    return outDataCol;
}


// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(vector_extension, in_data_f) { \
    new typename t_varex_t::operator_wrapper::template for_output_formats<uncompr_f>::template for_input_formats<in_data_f, uncompr_f>( \
        &simple_project<vector_extension, in_data_f> \
    ), \
    STR_EVAL_MACROS(vector_extension), \
    STR_EVAL_MACROS(in_data_f), \
}

template<class t_varex_t, unsigned t_Bw>
std::vector<typename t_varex_t::variant_t> make_variants() {
    return {
        // Reference variant.
        {
            new typename t_varex_t::operator_wrapper::template for_output_formats<uncompr_f>::template for_input_formats<uncompr_f, uncompr_f>(
                &copy
            ),
            "copy",
            "copy"
        },
        
        // Compressed variants.
        MAKE_VARIANT(scalar<v64<uint64_t>>, uncompr_f),
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 1>>)),
#ifdef SSE
        MAKE_VARIANT(sse<v128<uint64_t>>, uncompr_f),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 2>>)),
#endif
#ifdef AVXTWO
        MAKE_VARIANT(avx2<v256<uint64_t>>, uncompr_f),
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 4>>)),
#endif
#ifdef AVX512
        MAKE_VARIANT(avx512<v512<uint64_t>>, uncompr_f),
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 8>>)),
#endif
    };
}



// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 2>::type
        ::for_variant_params<std::string, std::string>
        ::for_setting_params<unsigned, size_t>;
    varex_t varex(
            {},
            {"vector_extension", "format"},
            {"bit width", "countDataLog"}
    );
    
    for(size_t countDataLog : {512, 512 * 1024}) {
        for(
                unsigned bw = effective_bitwidth(countDataLog - 1);
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
            auto origDataCol = generate_sorted_unique(countDataLog);
            auto origPosCol = generate_sorted_unique(countDataLog);
            varex.print_datagen_done();

            varex.execute_variants(
                    variants, bw, countDataLog, origDataCol, origPosCol
            );

            delete origDataCol;
            delete origPosCol;
        }
    }
    
    varex.done();
    
    return !varex.good();
}