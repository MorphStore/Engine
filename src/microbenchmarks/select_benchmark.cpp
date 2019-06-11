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
#include <core/morphing/static_vbp.h>
#include <core/operators/general_vectorized/select_compr.h>
#include <core/operators/scalar/select_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/variant_executor.h>
#include <vector/primitives/compare.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/scalar/primitives/compare_scalar.h>

#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vector;


template<unsigned t_bw>
const column<static_vbp_f<t_bw, 1> > * foo_select(
        const column<uncompr_f> * const inDataCol,
        const uint64_t val,
        MSV_CXX_ATTRIBUTE_PPUNUSED const size_t outPosCountEstimate = 0
) {
    using out_f = static_vbp_f<t_bw, 1>;

    const size_t inDataCount = inDataCol->get_count_values();
    const uint64_t * const inData = inDataCol->get_data();

    // If no estimate is provided: Pessimistic allocation size (for
    // uncompressed data), reached only if all input data elements pass the
    // selection.
    auto outPosCol = new column<out_f>(out_f::get_size_max_byte(inDataCount));

    uint8_t * outPos = outPosCol->get_data();

    uint64_t outBuffer64[64];
    unsigned outBufferPos = 0;
    size_t countMatches = 0;

    for(unsigned i = 0; i < inDataCount; i++)
        if(inData[i] == val) {
            outBuffer64[outBufferPos++] = i;
            if(outBufferPos == 64) {
                const uint8_t * outBuffer8 = reinterpret_cast<const uint8_t *>(outBuffer64);
                pack<vector::scalar<vector::v64<uint64_t>>, t_bw, 1>(outBuffer8, 64, outPos);
                outBufferPos = 0;
                countMatches += 64;
            }
        }

    if(outBufferPos != 0)
        throw std::runtime_error("ohoh " + std::to_string(outBufferPos));

    outPosCol->set_meta_data(countMatches, out_f::get_size_max_byte(countMatches));

    return outPosCol;
}


// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(out_data_f, outDataFName, in_data_f, inDataFName, bw) { \
    new varex_t::operator_wrapper::for_output_formats<out_data_f>::for_input_formats<in_data_f>( \
        &morphstore::select<std::equal_to, scalar<v64<uint64_t>>, out_data_f, in_data_f> \
    ), \
    STR_EVAL_MACROS(outDataFName), \
    STR_EVAL_MACROS(inDataFName), \
    bw \
}

#define MAKE_VARIANTS(bw) \
    MAKE_VARIANT(uncompr_f, "uncompr_f", uncompr_f, "uncompr_f", bw), \
    { \
        new varex_t::operator_wrapper::for_output_formats<uncompr_f>::for_input_formats<static_vbp_f<bw, 1>>( \
            &my_select_t<vector::equal, scalar<v64<uint64_t>>, static_vbp_f<bw, 1>>::apply \
        ), \
        STR_EVAL_MACROS("uncompr_f"), \
        STR_EVAL_MACROS("static_vbp_f<bw, 1>"), \
        bw \
    }, \
    { \
        new varex_t::operator_wrapper::for_output_formats<static_vbp_f<bw, 1> >::for_input_formats<uncompr_f>( \
            &foo_select<bw> \
        ), \
        STR_EVAL_MACROS("static_vbp_f<bw, 1>"), \
        STR_EVAL_MACROS("uncompr_f"), \
        bw \
    }

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 1, uint64_t, size_t>::type
        ::for_variant_params<std::string, std::string, unsigned>
        ::for_setting_params<float>;
    varex_t varex(
            {"pred", "est"},
            {"out_data_f", "in_data_f", "bw"},
            {"selectivity"}
    );
    
    const size_t countValues = 128 * 1000 * 1000;
    const uint64_t pred = 0;
    
    for(float selectivity : {0.001, 0.01, 0.1, 0.5, 0.9}) 
        // The compressed output needs up to 27 bits per value. Thus, we cannot
        // start with less than 27 bits.
        for(unsigned bw = 27; bw <= std::numeric_limits<uint64_t>::digits; bw++) {
            std::vector<varex_t::variant_t> variants;
            switch(bw) {
                // Generated with python:
                // for bw in range(1, 64+1):
                //   print("            case {: >2}: variants = {{MAKE_VARIANTS({: >2})}}; break;".format(bw, bw))
//                case  1: variants = {MAKE_VARIANTS( 1)}; break;
//                case  2: variants = {MAKE_VARIANTS( 2)}; break;
//                case  3: variants = {MAKE_VARIANTS( 3)}; break;
//                case  4: variants = {MAKE_VARIANTS( 4)}; break;
//                case  5: variants = {MAKE_VARIANTS( 5)}; break;
//                case  6: variants = {MAKE_VARIANTS( 6)}; break;
//                case  7: variants = {MAKE_VARIANTS( 7)}; break;
//                case  8: variants = {MAKE_VARIANTS( 8)}; break;
//                case  9: variants = {MAKE_VARIANTS( 9)}; break;
//                case 10: variants = {MAKE_VARIANTS(10)}; break;
//                case 11: variants = {MAKE_VARIANTS(11)}; break;
//                case 12: variants = {MAKE_VARIANTS(12)}; break;
//                case 13: variants = {MAKE_VARIANTS(13)}; break;
//                case 14: variants = {MAKE_VARIANTS(14)}; break;
//                case 15: variants = {MAKE_VARIANTS(15)}; break;
//                case 16: variants = {MAKE_VARIANTS(16)}; break;
//                case 17: variants = {MAKE_VARIANTS(17)}; break;
//                case 18: variants = {MAKE_VARIANTS(18)}; break;
//                case 19: variants = {MAKE_VARIANTS(19)}; break;
//                case 20: variants = {MAKE_VARIANTS(20)}; break;
//                case 21: variants = {MAKE_VARIANTS(21)}; break;
//                case 22: variants = {MAKE_VARIANTS(22)}; break;
//                case 23: variants = {MAKE_VARIANTS(23)}; break;
//                case 24: variants = {MAKE_VARIANTS(24)}; break;
//                case 25: variants = {MAKE_VARIANTS(25)}; break;
//                case 26: variants = {MAKE_VARIANTS(26)}; break;
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
            const size_t countMatches = static_cast<size_t>(static_cast<float>(countValues) * selectivity) / 64 * 64;
            auto origCol = generate_exact_number(
                    countValues,
                    countMatches,
                    0,
                    bitwidth_max<uint64_t>(bw)
            );
            varex.print_datagen_done();

            varex.execute_variants(variants, selectivity, origCol, pred, 0);

            delete origCol;
        }
    
    varex.done();
    
    return 0;
}