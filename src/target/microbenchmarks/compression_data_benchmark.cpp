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
 * @file compression_data_benchmark.cpp
 * @brief Measures (de)compression runtimes on data sets with various data
 * characteristics.
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
#include <core/operators/general_vectorized/agg_sum_compr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/data_properties.h>
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

// @todo Don't duplicate this code from the calibration benchmarks.

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
        
        outCol->set_meta_data(
                p_DstCountValues, outData - initOutData, outData - initOutData
        );
        return outCol;
    }
};


// ****************************************************************************
// "Operator" to be executed by `variant_executor`
// ****************************************************************************

template<class t_vector_extension, class t_format>
const std::tuple<const column<uncompr_f> *, const column<uncompr_f> *> measure_morphs(
        const column<uncompr_f> * p_InCol, // small column
        size_t p_CountValuesLarge,
        size_t p_SettingGroup,
        size_t p_SettingIdx,
        int p_RepIdx,
        unsigned p_Bw
) {
    MSV_CXX_ATTRIBUTE_PPUNUSED
    const size_t countValuesSmall = p_InCol->get_count_values();
    
    // ------------------------------------------------------------------------
    // Record some data characteristics etc.
    // ------------------------------------------------------------------------
    
    MONITORING_ADD_INT_FOR(
            "countValuesSmall", countValuesSmall,
            veName<t_vector_extension>, formatName<t_format>, p_CountValuesLarge, p_SettingGroup, p_SettingIdx, p_RepIdx, p_Bw
    );
    MONITORING_ADD_DATAPROPERTIES_FOR(
            "", data_properties(p_InCol),
            veName<t_vector_extension>, formatName<t_format>, p_CountValuesLarge, p_SettingGroup, p_SettingIdx, p_RepIdx, p_Bw
    );
    
    // ------------------------------------------------------------------------
    // Measure runtimes.
    // ------------------------------------------------------------------------
    
    MONITORING_START_INTERVAL_FOR(
            "runtime compr cache2ram [µs]",
            veName<t_vector_extension>, formatName<t_format>, p_CountValuesLarge, p_SettingGroup, p_SettingIdx, p_RepIdx, p_Bw
    );
    // large column
    auto comprCol = cache2ram_morph_t<
            t_vector_extension, t_format, uncompr_f
    >::apply(p_InCol, p_CountValuesLarge);
    MONITORING_END_INTERVAL_FOR(
            "runtime compr cache2ram [µs]",
            veName<t_vector_extension>, formatName<t_format>, p_CountValuesLarge, p_SettingGroup, p_SettingIdx, p_RepIdx, p_Bw
    );
    
    MONITORING_START_INTERVAL_FOR(
            "runtime decompr ram2reg [µs]",
            veName<t_vector_extension>, formatName<t_format>, p_CountValuesLarge, p_SettingGroup, p_SettingIdx, p_RepIdx, p_Bw
    );
    // single-element column
    auto sumCol1 = agg_sum<t_vector_extension, t_format>(comprCol);
    MONITORING_END_INTERVAL_FOR(
            "runtime decompr ram2reg [µs]",
            veName<t_vector_extension>, formatName<t_format>, p_CountValuesLarge, p_SettingGroup, p_SettingIdx, p_RepIdx, p_Bw
    );
    
    MONITORING_START_INTERVAL_FOR(
            "runtime decompr ram2ram [µs]",
            veName<t_vector_extension>, formatName<t_format>, p_CountValuesLarge, p_SettingGroup, p_SettingIdx, p_RepIdx, p_Bw
    );
    // large column
    auto decomprCol1 = morph<t_vector_extension, uncompr_f>(comprCol);
    MONITORING_END_INTERVAL_FOR(
            "runtime decompr ram2ram [µs]",
            veName<t_vector_extension>, formatName<t_format>, p_CountValuesLarge, p_SettingGroup, p_SettingIdx, p_RepIdx, p_Bw
    );
    
    delete comprCol;
    
    MONITORING_START_INTERVAL_FOR(
            "runtime compr ram2ram [µs]",
            veName<t_vector_extension>, formatName<t_format>, p_CountValuesLarge, p_SettingGroup, p_SettingIdx, p_RepIdx, p_Bw
    );
    // large column
    auto comprCol2 = morph<t_vector_extension, t_format>(decomprCol1);
    MONITORING_END_INTERVAL_FOR(
            "runtime compr ram2ram [µs]",
            veName<t_vector_extension>, formatName<t_format>, p_CountValuesLarge, p_SettingGroup, p_SettingIdx, p_RepIdx, p_Bw
    );
    
    delete decomprCol1;
    
    // single-element column
    MSV_CXX_ATTRIBUTE_PPUNUSED
    auto sumCol2 = agg_sum<t_vector_extension, t_format>(comprCol2);
    
    delete comprCol2;
    
    return std::make_tuple(sumCol1, sumCol2);
}


// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(vector_extension, format) { \
    new typename t_varex_t::operator_wrapper::template for_output_formats<uncompr_f, uncompr_f>::template for_input_formats<uncompr_f>( \
        &measure_morphs<vector_extension, format>, true \
    ), \
    veName<vector_extension>, \
    formatName<format>, \
}

template<class t_varex_t, unsigned t_Bw>
std::vector<typename t_varex_t::variant_t> make_variants() {
    return {
#ifdef AVX512
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 8>>)),
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<512, 64, 8>)),
#elif defined(AVXTWO)
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 4>>)),
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<256, 32, 4>)),
#elif defined(SSE)
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 2>>)),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<128, 16, 2>)),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(k_wise_ns_f<2>), t_Bw),
#else
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 1>>)),
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<64, 8, 1>)),
#endif
    };
}


// ****************************************************************************
// Utilities for data generation.
// ****************************************************************************

#ifndef COMPRESSION_DATA_BENCHMARK_USE_HIST
class data_generator {
protected:
    virtual const column<uncompr_f> * generate_internal(
            size_t p_CountValues, bool p_IsSorted
    ) = 0;

    data_generator() {}

public:
    const column<uncompr_f> * generate(size_t p_CountValues, bool p_IsSorted) {
        return generate_internal(p_CountValues, p_IsSorted);
    }
    
    virtual ~data_generator() {
        //
    };
};

template<template<typename> class t_distr>
class special_data_generator : public data_generator {
protected:
    const t_distr<uint64_t> m_Distr;

    const column<uncompr_f> * generate_internal(
            size_t p_CountValues, bool p_IsSorted
    ) {
        return ColumnGenerator::generate_with_distr<t_distr>(
                p_CountValues, m_Distr, p_IsSorted
        );
    }
public:
    special_data_generator(t_distr<uint64_t> p_Distr) : m_Distr(p_Distr) {}
};

template<typename t_int_t>
using normal_int_distribution =
        int_distribution<std::normal_distribution<double>>::distr<t_int_t>;

template<typename t_int_t>
using two_normal_int_distribution = two_distr_distribution<
        normal_int_distribution, normal_int_distribution
>::distr<t_int_t>;

#endif

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(int argc, char ** argv) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    // @todo Get rid of the duplication here.
#ifdef COMPRESSION_DATA_BENCHMARK_USE_HIST
    if(argc != 5) {
        std::cerr
                << "Usage: " << argv[0]
                << " <countValuesLarge INT> <countValuesSmall INT> <repetitions INT> <bwWeightsFile STRING>" << std::endl
                << "countValuesLarge and countValuesSmall must be multiples of the number of data elements per vector." << std::endl;
        exit(-1);
    }
#else
    if(argc != 4) {
        std::cerr
                << "Usage: " << argv[0]
                << " <countValuesLarge INT> <countValuesSmall INT> <repetitions INT>" << std::endl
                << "countValuesLarge and countValuesSmall must be multiples of the number of data elements per vector." << std::endl;
        exit(-1);
    }
#endif
    // @todo More validation of the arguments.
    const size_t countValuesLarge = atoi(argv[1]);
    const size_t countValuesSmall = atoi(argv[2]);
    const int countRepetitions = atoi(argv[3]);
    if(countRepetitions < 1)
        throw std::runtime_error("the number of repetitions must be >= 1");
#ifdef COMPRESSION_DATA_BENCHMARK_USE_HIST
    const std::string bwWeightsFile(argv[4]);
#endif
    
    using varex_t = variant_executor_helper<2, 1, size_t, size_t, size_t, int, unsigned>::type
        ::for_variant_params<std::string, std::string>
        ::for_setting_params<>;
    varex_t varex(
            {"countValuesLarge", "settingGroup", "settingIdx", "repIdx", "bitwidth"},
            {"vector_extension", "format"},
            {}
    );
    
    const size_t digits = std::numeric_limits<uint64_t>::digits;
    
#ifdef COMPRESSION_DATA_BENCHMARK_USE_HIST
    // Read the weights of the bit width histograms from a file.
    const uint64_t * bwHists;
    size_t countBwHists;
    std::tie(bwHists, countBwHists) =
            generate_with_bitwidth_histogram_helpers::read_bw_weights(bwWeightsFile);
#else
    // Define the data distributions.
    std::vector<std::tuple<size_t, data_generator *>> generators;
    // Uniform distribution.
    for(unsigned bw = 1; bw <= digits; bw++)
        generators.push_back(
                {
                    1,
                    new special_data_generator<std::uniform_int_distribution>(
                            std::uniform_int_distribution<uint64_t>(
                                    0, bitwidth_max<uint64_t>(bw)
                            )
                    )
                }
        );
    // Normal distribution.
    for(unsigned startBw : {6, 21, 36, 51}) {
        const double stddev = bitwidth_max<uint64_t>(startBw) / 3;
        for(unsigned bw = startBw; bw < digits; bw++)
            generators.push_back(
                    {
                        2,
                        new special_data_generator<normal_int_distribution>(
                                normal_int_distribution<uint64_t>(
                                        std::normal_distribution<double>(
                                                bitwidth_max<uint64_t>(bw), stddev
                                        )
                                )
                        )
                    }
            );
    }
    // Two normal distributions.
    for(double outlierShare : {0.001, 0.01, 0.1, 0.25, 0.5, 0.9})
        for(unsigned startBw : {6, 21, 36, 51})
            for(unsigned bw = startBw; bw < digits; bw++) {
                const double stddev = bitwidth_max<uint64_t>(startBw) / 3;
                generators.push_back(
                        {
                            3,
                            new special_data_generator<two_normal_int_distribution>(
                                    two_normal_int_distribution<uint64_t>(
                                            normal_int_distribution<uint64_t>(
                                                    std::normal_distribution<double>(
                                                            bitwidth_max<uint64_t>(startBw),
                                                            stddev
                                                    )
                                            ),
                                            normal_int_distribution<uint64_t>(
                                                    std::normal_distribution<double>(
                                                            bitwidth_max<uint64_t>(bw),
                                                            stddev
                                                    )
                                            ),
                                            outlierShare
                                    )
                            )
                        }
                );
            }
#endif
    
    
    for(int repIdx = 1; repIdx <= countRepetitions; repIdx++) {
        size_t settingIdx = 0;
        
#ifdef COMPRESSION_DATA_BENCHMARK_USE_HIST
        for(unsigned bwHistIdx = 0; bwHistIdx < countBwHists; bwHistIdx++) {
            const uint64_t * const bwHist = bwHists + bwHistIdx * digits;
            
            unsigned maxBw = digits;
            while(!bwHist[maxBw - 1])
                maxBw--;
            
            size_t settingGroup = 0;
#else
        for(auto genInfo : generators) {
            size_t settingGroup;
            data_generator * generator;
            std::tie(settingGroup, generator) = genInfo;
#endif
            
            for(bool isSorted : {false, true}) {
                settingIdx++;

                varex.print_datagen_started();
#ifdef COMPRESSION_DATA_BENCHMARK_USE_HIST
                auto origCol = generate_with_bitwidth_histogram(
                        countValuesSmall, bwHist, isSorted, true
                );
#else
                // Generate the data.
                auto origCol = generator->generate(countValuesSmall, isSorted);
                // Find out the maximum bit width.
                uint64_t pseudoMax = 0;
                const uint64_t * origData = origCol->get_data();
                for(size_t i = 0; i < countValuesSmall; i++)
                    pseudoMax |= origData[i];
                const unsigned maxBw = effective_bitwidth(pseudoMax);
#endif
                varex.print_datagen_done();

                std::vector<varex_t::variant_t> variants;
                switch(maxBw) {
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

                varex.execute_variants(
                        variants,
                        origCol,
                        countValuesLarge,
                        settingGroup,
                        settingIdx,
                        repIdx, 
                        maxBw
                );

                delete origCol;
            }
        }
    }
     
#ifndef COMPRESSION_DATA_BENCHMARK_USE_HIST
    for(auto genInfo : generators)
        delete std::get<1>(genInfo);
#endif
    
    varex.done();
    
    return !varex.good();
}
