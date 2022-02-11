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
 * @file bw_prof_casc.cpp
 * @brief Measures the bit width profiles of (physical-level) Null Suppression
 * algorithms in cascade situations.
 * 
 * These are required for our cost model for lightweight compression 
 * algorithms.
 * 
 * The output is produced as a CSV table on stdout.
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/k_wise_ns.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/format_names.h> // Must be included after all formats.
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/variant_executor.h>
#include <vector/vector_extension_names.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cstdlib>

using namespace morphstore;
using namespace vectorlib;


// ****************************************************************************
// Variant of the morph-operator simulating in-cascade use
// ****************************************************************************

// ----------------------------------------------------------------------------
// Interface
// ----------------------------------------------------------------------------

/**
 * @brief A variant of the morph-operator simulating in-cascade use of the
 * morph-operator for (physical-level) Null Suppression formats.
 * 
 * That is, compressions must effectively read from cache and write to memory,
 * while decompressions must read from memory and write to cache.
 */
template<class t_vector_extension, class t_dst_f, class t_src_f>
struct cache2ram_morph_t {
    static const column<t_dst_f> * apply(
            const column<t_src_f> * p_InCol, size_t p_DstCountValues
    ) = delete;
};

// ----------------------------------------------------------------------------
// Compression
// ----------------------------------------------------------------------------

/**
 * @brief A variant of the morph-operator simulating in-cascade use of the
 * morph-operator for (physical-level) Null Suppression formats.
 * 
 * This variant effectively reads from cache and writes to memory by repeatedly
 * calling the compressing batch-level morph-operator until `p_DstCountValues`
 * logical data elements are represented in the compressed output. The input
 * is always the same (small) buffer, namely the data buffer of the (small)
 * input column. However, the output pointer moves on with every call, so that
 * a large output is generated.
 */
template<class t_vector_extension, class t_dst_f>
struct cache2ram_morph_t<t_vector_extension, t_dst_f, uncompr_f> {
    static const column<t_dst_f> * apply(
            const column<uncompr_f> * p_InCol, size_t p_DstCountValues
    ) {
        const size_t srcCountValues = p_InCol->get_count_values();
        
        if(srcCountValues % t_dst_f::m_BlockSize)
            throw std::runtime_error(
                    "cache2ram_morph_t compression: the number of data "
                    "elements in the input column must be a multiple of the "
                    "destination format's block size"
            );
        if(p_DstCountValues % srcCountValues)
            throw std::runtime_error(
                    "cache2ram_morph_t compression: the number of data "
                    "elements for the output column must be a multiple of the "
                    "number of data elements in the input column"
            );
        
        auto outCol = new column<t_dst_f>(
                t_dst_f::get_size_max_byte(p_DstCountValues)
        );
        uint8_t * outData = outCol->get_data();
        const uint8_t * const initOutData = outData;
        
        size_t countCompressed = 0;
        while(countCompressed < p_DstCountValues) {
            const uint8_t * inData = p_InCol->get_data();
            morph_batch<t_vector_extension, t_dst_f, uncompr_f>(
                    inData, outData, srcCountValues
            );
            countCompressed += srcCountValues;
        }
        
        outCol->set_meta_data(p_DstCountValues, outData - initOutData);
        return outCol;
    }
};

// ----------------------------------------------------------------------------
// Decompression
// ----------------------------------------------------------------------------

/**
 * @brief A variant of the morph-operator simulating in-cascade use of the
 * morph-operator for (physical-level) Null Suppression formats.
 * 
 * This variant effectively reads from memory and writes to cache by repeatedly
 * calling the decompressing batch-level morph-operator until all logical data
 * elements in the (large) input column have been consumed. The output is
 * always the same (small) buffer, namely the data buffer of the (small) output
 * column. However, the input pointer moves on with every call, so that only a
 * small output is generated.
 */
template<class t_vector_extension, class t_src_f>
struct cache2ram_morph_t<t_vector_extension, uncompr_f, t_src_f> {
    static const column<uncompr_f> * apply(
            const column<t_src_f> * p_InCol, size_t p_DstCountValues
    ) {
        const size_t srcCountValues = p_InCol->get_count_values();
        
        if(p_DstCountValues % t_src_f::m_BlockSize)
            throw std::runtime_error(
                    "cache2ram_morph_t decompression: the number of data "
                    "elements in the output column must be a multiple of the "
                    "source format's block size"
            );
        if(srcCountValues % p_DstCountValues)
            throw std::runtime_error(
                    "cache2ram_morph_t decompression: the number of data "
                    "elements in th input column must be a multiple of the "
                    "number of data elements in the output column"
            );
        
        const uint8_t * inData = p_InCol->get_data();
        
        const size_t outSizeAlloc =
                uncompr_f::get_size_max_byte(p_DstCountValues);
        auto outCol = new column<uncompr_f>(outSizeAlloc);
        
        size_t countDecompressed = 0;
        while(countDecompressed < srcCountValues) {
            uint8_t * outData = outCol->get_data();
            morph_batch<t_vector_extension, uncompr_f, t_src_f>(
                    inData, outData, p_DstCountValues
            );
            countDecompressed += p_DstCountValues;
        }
        
        outCol->set_meta_data(p_DstCountValues, outSizeAlloc);
        return outCol;
    }
};


// ****************************************************************************
// "Operator" to be executed by `variant_executor`
// ****************************************************************************

/**
 * @brief Measures the compression and decompression time for the specified
 * format.
 * @param p_InCol The column to be compressed and decompressed. This column is
 * assumed to be small.
 * @return The decompressed-again input column.
 */
template<class t_vector_extension, class t_format, unsigned t_Bw>
const column<uncompr_f> * measure_morphs(
        const column<uncompr_f> * p_InCol,
        size_t p_CountValuesLarge,
        int p_RepIdx
) {
    const size_t countValuesSmall = p_InCol->get_count_values();
    
    MONITORING_START_INTERVAL_FOR(
            "runtime compr [µs]",
            veName<t_vector_extension>, formatName<t_format>, t_Bw, countValuesSmall, p_CountValuesLarge, p_RepIdx
    );
    auto comprCol = cache2ram_morph_t<
            t_vector_extension, t_format, uncompr_f
    >::apply(p_InCol, p_CountValuesLarge);
    MONITORING_END_INTERVAL_FOR(
            "runtime compr [µs]",
            veName<t_vector_extension>, formatName<t_format>, t_Bw, countValuesSmall, p_CountValuesLarge, p_RepIdx
    );
            
    MONITORING_START_INTERVAL_FOR(
            "runtime decompr [µs]",
            veName<t_vector_extension>, formatName<t_format>, t_Bw, countValuesSmall, p_CountValuesLarge, p_RepIdx
    );
    auto decomprCol = cache2ram_morph_t<
            t_vector_extension, uncompr_f, t_format
    >::apply(comprCol, countValuesSmall);
    MONITORING_END_INTERVAL_FOR(
            "runtime decompr [µs]",
            veName<t_vector_extension>, formatName<t_format>, t_Bw, countValuesSmall, p_CountValuesLarge, p_RepIdx
    );
    
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
    // Although static_vbp_f is not intended to be used in cascades, we need
    // its profile anyway for the operator cost model.
    return {
#ifdef AVX512
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<512, 64, 8>), t_Bw),
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 8>>), t_Bw),
#elif defined(AVXTWO)
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<256, 32, 4>), t_Bw),
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 4>>), t_Bw),
#elif defined(SSE)
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<128, 16, 2>), t_Bw),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(k_wise_ns_f<2>), t_Bw),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 2>>), t_Bw),
#else
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<64, 8, 1>), t_Bw),
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 1>>), t_Bw),
#endif
    };
}


// ****************************************************************************
// Main program.
// ****************************************************************************

int main(int argc, char ** argv) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    if(argc != 2)
        throw std::runtime_error(
                "this calibration benchmark expects the number of repetitions "
                "as its only argument"
        );
    const int countRepetitions = atoi(argv[1]);
    if(countRepetitions < 1)
        throw std::runtime_error("the number of repetitions must be >= 1");
    
    // @todo Actually, the repetition number should be a setting parameter.
    // However, we need it in the variants to add more measurements.
    using varex_t = variant_executor_helper<1, 1, size_t, int>::type
        ::for_variant_params<std::string, std::string>
        ::for_setting_params<unsigned, size_t>;
    varex_t varex(
            {"countValuesLarge", "repetition"},
            {"vector_extension", "format"},
            {"bitwidth", "countValuesSmall"}
    );
    
    // @todo These could be command line arguments.
    const size_t countValuesLarge = 128 * 1024 * 1024;
    // Cascade block size in terms of logical data elements.
    const size_t countValuesSmall = 1024;
    
    for(int repIdx = 1; repIdx <= countRepetitions; repIdx++)
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
            auto origCol = ColumnGenerator::generate_with_distr(
                    countValuesSmall,
                    std::uniform_int_distribution<uint64_t>(
                            bitwidth_min<uint64_t>(bw), bitwidth_max<uint64_t>(bw)
                    ),
                    false
            );
            varex.print_datagen_done();

            varex.execute_variants(
                    variants,
                    bw, countValuesSmall,
                    origCol,
                    countValuesLarge, repIdx
            );

            delete origCol;
        }
    
    varex.done();
    
    return !varex.good();
}
