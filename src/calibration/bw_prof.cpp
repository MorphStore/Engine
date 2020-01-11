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
 * @file bw_prof.cpp
 * @brief Measures the bit width profiles of (physical-level) Null Suppression
 * algorithms required for our cost model for lightweight compression
 * algorithms.
 * 
 * The output is produced as a CSV table on stdout.
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/k_wise_ns.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/vbp.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <core/utils/variant_executor.h>
#include <vector/vector_extension_names.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <limits>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>


using namespace morphstore;
using namespace vectorlib;


// ****************************************************************************
// Mapping from formats to string names
// ****************************************************************************

// All template-specializations of a format are mapped to a name, which may or
// may not contain the values of the template parameters.
        
template<class t_format>
std::string formatName = "(unknown format)";

template<size_t t_BlockSizeLog, size_t t_PageSizeBlocks, unsigned t_Step>
std::string formatName<
        dynamic_vbp_f<t_BlockSizeLog, t_PageSizeBlocks, t_Step>
> = "dynamic_vbp_f<" + std::to_string(t_BlockSizeLog) + ", " + std::to_string(t_PageSizeBlocks) + ", " + std::to_string(t_Step) + ">";

template<size_t t_BlockSizeLog>
std::string formatName<k_wise_ns_f<t_BlockSizeLog>> = "k_wise_ns_f<" + std::to_string(t_BlockSizeLog) + ">";

template<unsigned t_Bw, unsigned t_Step>
std::string formatName<
        static_vbp_f<vbp_l<t_Bw, t_Step> >
> = "static_vbp_f<vbp_l<bw, " + std::to_string(t_Step) + "> >";

template<>
std::string formatName<uncompr_f> = "uncompr_f";


// ****************************************************************************
// "Operator" to be executed by `variant_executor`
// ****************************************************************************

/**
 * @brief Measures the compression and decompression time as well as the
 * compressed size for the specified format.
 * @param p_InCol The column to be compressed and decompressed.
 * @return The decompressed-again input column.
 */
template<class t_vector_extension, class t_format, unsigned t_Bw>
const column<uncompr_f> * measure_morphs(const column<uncompr_f> * p_InCol) {
    // This is unused iff monitoring is disabled.
    MSV_CXX_ATTRIBUTE_PPUNUSED const size_t countValues =
        p_InCol->get_count_values();
    
    MONITORING_START_INTERVAL_FOR(
            "runtime compr [µs]",
            veName<t_vector_extension>, formatName<t_format>, t_Bw, countValues
    );
    auto comprCol = morph<t_vector_extension, t_format, uncompr_f>(p_InCol);
    MONITORING_END_INTERVAL_FOR(
            "runtime compr [µs]",
            veName<t_vector_extension>, formatName<t_format>, t_Bw, countValues
    );
            
    MONITORING_ADD_INT_FOR(
            "size used [byte]",
            comprCol->get_size_used_byte(),
            veName<t_vector_extension>, formatName<t_format>, t_Bw, countValues
    );
    MONITORING_ADD_INT_FOR(
            "size compr [byte]",
            comprCol->get_size_compr_byte(),
            veName<t_vector_extension>, formatName<t_format>, t_Bw, countValues
    );
            
    MONITORING_START_INTERVAL_FOR(
            "runtime decompr [µs]",
            veName<t_vector_extension>, formatName<t_format>, t_Bw, countValues
    );
    auto decomprCol = morph<t_vector_extension, uncompr_f, t_format>(comprCol);
    MONITORING_END_INTERVAL_FOR(
            "runtime decompr [µs]",
            veName<t_vector_extension>, formatName<t_format>, t_Bw, countValues
    );
    
    if(!std::is_same<t_format, uncompr_f>::value)
        delete comprCol;
    
    return decomprCol;
}


// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(vector_extension, format, bw) { \
    new typename t_varex_t::operator_wrapper::template for_output_formats<uncompr_f>::template for_input_formats<uncompr_f>( \
        &measure_morphs<vector_extension, format, bw>, true \
    ), \
    veName<vector_extension>, \
    formatName<format>, \
}

template<class t_varex_t, unsigned t_Bw>
std::vector<typename t_varex_t::variant_t> make_variants() {
    return {
        // Uncompressed reference variant, required to check the correctness of
        // the compressed variants.
        MAKE_VARIANT(scalar<v64<uint64_t>>, uncompr_f, t_Bw),
        
        // Compressed variants.
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<64, 8, 1>), t_Bw),
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 1>>), t_Bw),
#ifdef SSE
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<128, 16, 2>), t_Bw),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(k_wise_ns_f<2>), t_Bw),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 2>>), t_Bw),
#endif
#ifdef AVXTWO
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<256, 32, 4>), t_Bw),
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 4>>), t_Bw),
#endif
#ifdef AVX512
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<512, 64, 8>), t_Bw),
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 8>>), t_Bw),
#endif
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
    
    // @todo This could be a command line argument.
    const size_t countValues = 128 * 1024 * 1024;
    
    for(unsigned bw = 1; bw <= std::numeric_limits<uint64_t>::digits; bw++) {
        std::vector<varex_t::variant_t> variants;
        switch(bw) {
            // Generated with python:
            // for bw in range(1, 64+1):
            //   print("            case {: >2}: variants = make_variants<varex_t, {: >2}>(); break;".format(bw, bw))
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
        auto origCol = generate_with_distr(
                countValues,
                std::uniform_int_distribution<uint64_t>(
                        bitwidth_min<uint64_t>(bw), bitwidth_max<uint64_t>(bw)
                ),
                false
        );
        varex.print_datagen_done();

        varex.execute_variants(variants, bw, countValues, origCol);

//            delete origCol;
    }
    
    varex.done();
    
    return !varex.good();
}