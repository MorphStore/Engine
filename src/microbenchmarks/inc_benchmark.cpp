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
 * @file inc_benchmark.cpp
 * @brief A little mirco benchmark of an increment operator (which is none of
 * MorphStore's standard operators, but exists only in this file) on
 * uncompressed and compressed data.
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/static_vbp.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/variant_executor.h>

#include <vector/scalar/extension_scalar.h>
#include <vector/scalar/primitives/calc_scalar.h>
#include <vector/scalar/primitives/create_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/scalar/primitives/logic_scalar.h>
#ifdef AVXTWO
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/logic_avx2.h>
#endif
#ifdef AVX512
#include <vector/simd/avx512/extension_avx512.h>
#include <vector/simd/avx512/primitives/calc_avx512.h>
#include <vector/simd/avx512/primitives/create_avx512.h>
#include <vector/simd/avx512/primitives/io_avx512.h>
#include <vector/simd/avx512/primitives/logic_avx512.h>
#endif
#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/logic_sse.h>

#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vector;


template<class t_out_data_f, class t_in_data_f>
struct inc_t {
    static const column<t_out_data_f> * apply(const column<t_in_data_f> * p_InDataCol);
};

template<>
struct inc_t<uncompr_f, uncompr_f> {
    static const column<uncompr_f> * apply(const column<uncompr_f> * p_InDataCol) {
        const uint64_t * inData = p_InDataCol->get_data();
        const size_t inDataCount = p_InDataCol->get_count_values();

        auto outDataCol = new column<uncompr_f>(p_InDataCol->get_size_used_byte());
        uint64_t * outData = outDataCol->get_data();

        for(unsigned i = 0; i < inDataCount; i++)
            outData[i] = inData[i] + 1;

        outDataCol->set_meta_data(inDataCount, p_InDataCol->get_size_used_byte());
        return outDataCol;
    }
};

template<unsigned t_bw>
struct inc_t<uncompr_f, static_vbp_f<t_bw, 1> > {
    using out_f = uncompr_f;
    using in_f = static_vbp_f<t_bw, 1>;
    static const column<out_f> * apply(const column<in_f> * p_InDataCol) {
        const uint8_t * inData = p_InDataCol->get_data();
        const size_t inDataCount = p_InDataCol->get_count_values();

        auto outDataCol = new column<out_f>(out_f::get_size_max_byte(inDataCount));
        uint64_t * outData = outDataCol->get_data();

        uint64_t inBuffer64[64];

        for(unsigned i = 0; i < inDataCount; i += 64) {
            uint8_t * inBuffer8 = reinterpret_cast<uint8_t *>(inBuffer64);
            unpack<vector::scalar<vector::v64<uint64_t>>, t_bw, 1>(inData, inBuffer8, 64);
            for(unsigned k = 0; k < 64; k++)
                outData[i + k] = inBuffer64[k] + 1;
        }

        outDataCol->set_meta_data(inDataCount, out_f::get_size_max_byte(inDataCount));
        return outDataCol;
    }
};

template<unsigned t_bw>
struct inc_t<static_vbp_f<t_bw, 1>, uncompr_f> {
    using out_f = static_vbp_f<t_bw, 1>;
    using in_f = uncompr_f;
    static const column<out_f> * apply(const column<in_f> * p_InDataCol) {
        const uint64_t * inData = p_InDataCol->get_data();
        const size_t inDataCount = p_InDataCol->get_count_values();

        auto outDataCol = new column<out_f>(out_f::get_size_max_byte(inDataCount));
        uint8_t * outData = outDataCol->get_data();

        uint64_t outBuffer64[64];

        for(unsigned i = 0; i < inDataCount; i += 64) {
            for(unsigned k = 0; k < 64; k++)
                outBuffer64[k] = inData[i + k] + 1;
            const uint8_t * outBuffer8 = reinterpret_cast<uint8_t *>(outBuffer64);
            pack<vector::scalar<vector::v64<uint64_t>>, t_bw, 1>(outBuffer8, 64, outData);
        }

        outDataCol->set_meta_data(inDataCount, out_f::get_size_max_byte(inDataCount));
        return outDataCol;
    }
};

template<unsigned t_bw>
struct inc_t<static_vbp_f<t_bw, 1>, static_vbp_f<t_bw, 1> > {
    using out_f = static_vbp_f<t_bw, 1>;
    using in_f = static_vbp_f<t_bw, 1>;
    static const column<out_f> * apply(const column<in_f> * p_InDataCol) {
        const uint8_t * inData = p_InDataCol->get_data();
        const size_t inDataCount = p_InDataCol->get_count_values();

        auto outDataCol = new column<out_f>(out_f::get_size_max_byte(inDataCount));
        uint8_t * outData = outDataCol->get_data();

        uint64_t inBuffer64[64];
        uint64_t outBuffer64[64];

        for(unsigned i = 0; i < inDataCount; i += 64) {
            uint8_t * inBuffer8 = reinterpret_cast<uint8_t *>(inBuffer64);
            unpack<vector::scalar<vector::v64<uint64_t>>, t_bw, 1>(inData, inBuffer8, 64);
            for(unsigned k = 0; k < 64; k++)
                outBuffer64[k] = inBuffer64[k] + 1;
            const uint8_t * outBuffer8 = reinterpret_cast<const uint8_t *>(outBuffer64);
            pack<vector::scalar<vector::v64<uint64_t>>, t_bw, 1>(outBuffer8, 64, outData);
        }

        outDataCol->set_meta_data(inDataCount, out_f::get_size_max_byte(inDataCount));
        return outDataCol;
    }
};



// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(out_data_f, nameOutDataF, in_data_f, nameInDataF, bw) { \
    new varex_t::operator_wrapper::for_output_formats<out_data_f>::for_input_formats<in_data_f>( \
        &inc_t<out_data_f, in_data_f>::apply \
    ), \
    nameOutDataF, \
    nameInDataF, \
    bw \
}

#define MAKE_VARIANTS(bw) \
    MAKE_VARIANT(uncompr_f, "uncompr_f", uncompr_f, "uncompr_f", bw), \
    MAKE_VARIANT(uncompr_f, "uncompr_f", SINGLE_ARG(static_vbp_f<bw, 1>), "static_vbp_f<bw, 1>", bw), \
    MAKE_VARIANT(SINGLE_ARG(static_vbp_f<bw, 1>), "static_vbp_f<bw, 1>", uncompr_f, "uncompr_f", bw), \
    MAKE_VARIANT(SINGLE_ARG(static_vbp_f<bw, 1>), "static_vbp_f<bw, 1>", SINGLE_ARG(static_vbp_f<bw, 1>), "static_vbp_f<bw, 1>", bw)

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 1>::type
        ::for_variant_params<std::string, std::string, unsigned>
        ::for_setting_params<>;
    varex_t varex({}, {"out_data_f", "in_data_f", "bw"}, {});
    
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
                        0, bitwidth_max<uint64_t>(bw) - 1
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