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
 * @file agg_sum_benchmark.cpp
 * @brief A little mirco benchmark of the whole-column agg_sum operator on
 * uncompressed and compressed data.
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/uncompr.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <core/operators/general_vectorized/agg_sum_compr.h>
//#include <core/operators/scalar/agg_sum_compr_iterator.h>
//#include <core/operators/scalar/agg_sum_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/variant_executor.h>
#include <vector/scalar/extension_scalar.h>

#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>


// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(ps,in_data_f, inDataFName,bw) { \
    new varex_t::operator_wrapper::for_output_formats<uncompr_f>::for_input_formats<in_data_f>( \
        &agg_sum<ps, in_data_f> \
    ), \
    STR_EVAL_MACROS(ps), \
    STR_EVAL_MACROS(inDataFName), \
    bw \
}

#define MAKE_VARIANTS(bw) \
    MAKE_VARIANT(scalar<v64<uint64_t>>,SINGLE_ARG(static_vbp_f<bw, 1>), "static_vbp_f<bw, 1>",bw), \
    MAKE_VARIANT(sse<v128<uint64_t>>,SINGLE_ARG(static_vbp_f<bw, 2>), "static_vbp_f<bw, 1>",bw), \
    MAKE_VARIANT(avx2<v256<uint64_t>>,SINGLE_ARG(static_vbp_f<bw, 4>), "static_vbp_f<bw, 1>",bw),/* \
    MAKE_VARIANT(avx512<v512<uint64_t>>,SINGLE_ARG(static_vbp_f<bw, 8>), "static_vbp_f<bw, 1>",bw)
    */
    

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    
    using namespace morphstore;
    using namespace vectorlib;
    // @todo This should not be necessary.
    //fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 1>::type
        ::for_variant_params<std::string, std::string, unsigned>
        ::for_setting_params<>;
    varex_t varex({}, {"ps","in_data_f", "bw"}, {});
    
    const size_t countValues = 128 * 1000 * 1000;
    
    for(unsigned bw = 1; bw <= std::numeric_limits<uint64_t>::digits; bw++) {
        std::vector<varex_t::variant_t> variants;
        switch(bw) {
            // Generated with python:
            // for bw in range(1, 64+1):
            //   print("            case {: >2}: variants = {{MAKE_VARIANTS({: >2})}}; break;".format(bw, bw))
            case  1: variants = {MAKE_VARIANTS( 1)}; break;
            case  2: variants = {MAKE_VARIANTS( 2)}; break;
            case  3: variants = {MAKE_VARIANTS( 3)}; break;
            case  4: variants = {MAKE_VARIANTS( 4)}; break;
            case  5: variants = {MAKE_VARIANTS( 5)}; break;
            case  6: variants = {MAKE_VARIANTS( 6)}; break;
            case  7: variants = {MAKE_VARIANTS( 7)}; break;
            case  8: variants = {MAKE_VARIANTS( 8)}; break;
            case  9: variants = {MAKE_VARIANTS( 9)}; break;
            case 10: variants = {MAKE_VARIANTS(10)}; break;
            case 11: variants = {MAKE_VARIANTS(11)}; break;
            case 12: variants = {MAKE_VARIANTS(12)}; break;
            case 13: variants = {MAKE_VARIANTS(13)}; break;
            case 14: variants = {MAKE_VARIANTS(14)}; break;
            case 15: variants = {MAKE_VARIANTS(15)}; break;
            case 16: variants = {MAKE_VARIANTS(16)}; break;
            case 17: variants = {MAKE_VARIANTS(17)}; break;
            case 18: variants = {MAKE_VARIANTS(18)}; break;
            case 19: variants = {MAKE_VARIANTS(19)}; break;
            case 20: variants = {MAKE_VARIANTS(20)}; break;
            case 21: variants = {MAKE_VARIANTS(21)}; break;
            case 22: variants = {MAKE_VARIANTS(22)}; break;
            case 23: variants = {MAKE_VARIANTS(23)}; break;
            case 24: variants = {MAKE_VARIANTS(24)}; break;
            case 25: variants = {MAKE_VARIANTS(25)}; break;
            case 26: variants = {MAKE_VARIANTS(26)}; break;
            case 27: variants = {MAKE_VARIANTS(27)}; break;
            case 28: variants = {MAKE_VARIANTS(28)}; break;
            case 29: variants = {MAKE_VARIANTS(29)}; break;
            case 30: variants = {MAKE_VARIANTS(30)}; break;
            case 31: variants = {MAKE_VARIANTS(31)}; break;
            case 32: variants = {MAKE_VARIANTS(32)}; break;
            case 33: variants = {MAKE_VARIANTS(33)}; break;
            case 34: variants = {MAKE_VARIANTS(34)}; break;
            case 35: variants = {MAKE_VARIANTS(35)}; break;
            case 36: variants = {MAKE_VARIANTS(36)}; break;
            case 37: variants = {MAKE_VARIANTS(37)}; break;
            case 38: variants = {MAKE_VARIANTS(38)}; break;
            case 39: variants = {MAKE_VARIANTS(39)}; break;
            case 40: variants = {MAKE_VARIANTS(40)}; break;
            case 41: variants = {MAKE_VARIANTS(41)}; break;
            case 42: variants = {MAKE_VARIANTS(42)}; break;
            case 43: variants = {MAKE_VARIANTS(43)}; break;
            case 44: variants = {MAKE_VARIANTS(44)}; break;
            case 45: variants = {MAKE_VARIANTS(45)}; break;
            case 46: variants = {MAKE_VARIANTS(46)}; break;
            case 47: variants = {MAKE_VARIANTS(47)}; break;
            case 48: variants = {MAKE_VARIANTS(48)}; break;
            case 49: variants = {MAKE_VARIANTS(49)}; break;
            case 50: variants = {MAKE_VARIANTS(50)}; break;
            case 51: variants = {MAKE_VARIANTS(51)}; break;
            case 52: variants = {MAKE_VARIANTS(52)}; break;
            case 53: variants = {MAKE_VARIANTS(53)}; break;
            case 54: variants = {MAKE_VARIANTS(54)}; break;
            case 55: variants = {MAKE_VARIANTS(55)}; break;
            case 56: variants = {MAKE_VARIANTS(56)}; break;
            case 57: variants = {MAKE_VARIANTS(57)}; break;
            case 58: variants = {MAKE_VARIANTS(58)}; break;
            case 59: variants = {MAKE_VARIANTS(59)}; break;
            case 60: variants = {MAKE_VARIANTS(60)}; break;
            case 61: variants = {MAKE_VARIANTS(61)}; break;
            case 62: variants = {MAKE_VARIANTS(62)}; break;
            case 63: variants = {MAKE_VARIANTS(63)}; break;
            case 64: variants = {MAKE_VARIANTS(64)}; break;
        }
        
        varex.print_datagen_started();
        auto origCol = generate_with_distr(
                countValues,
                std::uniform_int_distribution<uint64_t>(
                        0, bitwidth_max<uint64_t>(bw)
                ),
                false
        );
        varex.print_datagen_done();

        varex.execute_variants(variants, origCol);
        
        delete origCol;
    }
    
    varex.done();
    
    return 0;
}