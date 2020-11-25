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
 * characteristics or simply analyzes the characteristics of the input data.
 * 
 * The output is produced as a CSV table on stdout.
 * 
 * What this microbenchmark actually does is controlled by two macro constants
 * which are set in CMakeLists.txt depending on the target name.
 * - COMPRESSION_DATA_BENCHMARK_DATA_SOURCE: Determines how the input data is
 *   obtained. It can be
 *   - generated to follow a given bit width histogram (DATA_SOURCE_HIST)
 *   - generated using some random distributions (DATA_SOURCE_DISTR)
 *   - loaded from a file (DATA_SOURCE_DATAFILE)
 * - COMPRESSION_DATA_BENCHMARK_RUN:
 *   - If defined: A couple of compression algorithms are run on the obtained
 *     input data and their runtimes and achieved compressed sizes are
 *     recorded.
 *   - If not defined: Only the data characteristics of the obtained input data
 *     are recorded.
 *   - Usually, both variants should be executed. Their CSV outputs can be
 *     joined on the columns ["settingGroup", "settingIdx", "countValuesSmall"].
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/monitoring.h>
#ifdef COMPRESSION_DATA_BENCHMARK_RUN
#include <core/morphing/format.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/group_simple.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/k_wise_ns.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/format_names.h> // Must be included after all formats.
#include <core/operators/general_vectorized/agg_sum_compr.h>
#include <core/utils/variant_executor.h>
#include <vector/vector_extension_names.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#else
#include <core/utils/data_properties.h>
#endif

#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cstdlib>

using namespace morphstore;
using namespace vectorlib;

// These values are used in CMakeLists.txt, keep them consistent.
#define DATA_SOURCE_HIST 1
#define DATA_SOURCE_DISTR 2
#define DATA_SOURCE_DATAFILE 3

#ifdef COMPRESSION_DATA_BENCHMARK_RUN

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
        MSV_CXX_ATTRIBUTE_PPUNUSED size_t p_SettingGroup,
        MSV_CXX_ATTRIBUTE_PPUNUSED size_t p_SettingIdx,
        MSV_CXX_ATTRIBUTE_PPUNUSED int p_RepIdx,
        MSV_CXX_ATTRIBUTE_PPUNUSED unsigned p_Bw
) {
    MSV_CXX_ATTRIBUTE_PPUNUSED
    const size_t countValuesSmall = p_InCol->get_count_values();
    
    // ------------------------------------------------------------------------
    // Record the data characteristics of the small column
    // ------------------------------------------------------------------------
    
    MONITORING_ADD_INT_FOR(
            "countValuesSmall", countValuesSmall,
            veName<t_vector_extension>, formatName<t_format>, p_CountValuesLarge, p_SettingGroup, p_SettingIdx, p_RepIdx, p_Bw
    );
    
    // ------------------------------------------------------------------------
    // Record compressed size of the small column
    // ------------------------------------------------------------------------
    
    auto comprColSmall = morph<t_vector_extension, t_format>(p_InCol);
    MONITORING_ADD_INT_FOR(
            "compressed size [byte]", comprColSmall->get_size_used_byte(),
            veName<t_vector_extension>, formatName<t_format>, p_CountValuesLarge, p_SettingGroup, p_SettingIdx, p_RepIdx, p_Bw
    );
    delete comprColSmall;
    
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
#if 1
        // Use only the allowed processing style with the largest vectors.
#ifdef AVX512
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 8>>)),
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<512, 64, 8>)),
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(group_simple_f<8, uint64_t, 64>)),
#elif defined(AVXTWO)
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 4>>)),
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<256, 32, 4>)),
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(group_simple_f<4, uint64_t, 32>)),
#elif defined(SSE)
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 2>>)),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<128, 16, 2>)),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(k_wise_ns_f<2>)),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(group_simple_f<2, uint64_t, 16>)),
#else
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 1>>)),
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<64, 8, 1>)),
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(group_simple_f<1, uint64_t, 8>)),
#endif
#else
        // Use all allowed processing styles.
#ifdef AVX512
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 8>>)),
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<512, 64, 8>)),
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(group_simple_f<8, uint64_t, 64>)),
#endif
#ifdef AVXTWO
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 4>>)),
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<256, 32, 4>)),
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(group_simple_f<4, uint64_t, 32>)),
#endif
#ifdef SSE
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 2>>)),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<128, 16, 2>)),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(k_wise_ns_f<2>)),
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(group_simple_f<2, uint64_t, 16>)),
#endif
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<t_Bw, 1>>)),
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<64, 8, 1>)),
        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(group_simple_f<1, uint64_t, 8>)),
#endif
    };
}

#endif

// ****************************************************************************
// Utilities for data generation.
// ****************************************************************************

#if COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_DISTR
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
        return generate_with_distr<t_distr>(
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
#if COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_HIST
#ifdef COMPRESSION_DATA_BENCHMARK_RUN
    if(argc != 5) {
        std::cerr
                << "Usage: " << argv[0]
                << " <countValuesLarge INT> <countValuesSmall INT> <repetitions INT> <bwWeightsFile STRING>" << std::endl
                << "countValuesLarge and countValuesSmall must be multiples of the number of data elements per vector." << std::endl;
        exit(-1);
    }
#else
    if(argc != 3) {
        std::cerr
                << "Usage: " << argv[0]
                << " <countValuesSmall INT> <bwWeightsFile STRING>" << std::endl
                << "countValuesSmall must be a multiple of the number of data elements per vector." << std::endl;
        exit(-1);
    }
#endif
#elif COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_DISTR
#ifdef COMPRESSION_DATA_BENCHMARK_RUN
    if(argc != 4) {
        std::cerr
                << "Usage: " << argv[0]
                << " <countValuesLarge INT> <countValuesSmall INT> <repetitions INT>" << std::endl
                << "countValuesLarge and countValuesSmall must be multiples of the number of data elements per vector." << std::endl;
        exit(-1);
    }
#else
    if(argc != 2) {
        std::cerr
                << "Usage: " << argv[0]
                << " <countValuesSmall INT>" << std::endl
                << "countValuesSmall must be a multiple of the number of data elements per vector." << std::endl;
        exit(-1);
    }
#endif
#elif COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_DATAFILE
#ifdef COMPRESSION_DATA_BENCHMARK_RUN
    if(argc != 5) {
        std::cerr
                << "Usage: " << argv[0]
                << " <countValuesLarge INT> <countValuesSmall INT> <repetitions INT> <dataFile STRING>" << std::endl
                << "countValuesLarge and countValuesSmall must be multiples of the number of data elements per vector." << std::endl;
        exit(-1);
    }
#else
    if(argc != 3) {
        std::cerr
                << "Usage: " << argv[0]
                << " <countValuesSmall INT> <dataFile STRING>" << std::endl
                << "countValuesSmall must be a multiple of the number of data elements per vector." << std::endl;
        exit(-1);
    }
#endif
#endif
    // @todo More validation of the arguments.
    unsigned argIdx = 1;
#ifdef COMPRESSION_DATA_BENCHMARK_RUN
    const size_t countValuesLarge = atoi(argv[argIdx++]);
#endif
    const size_t countValuesSmall = atoi(argv[argIdx++]);
#ifdef COMPRESSION_DATA_BENCHMARK_RUN
    const int countRepetitions = atoi(argv[argIdx++]);
    if(countRepetitions < 1)
        throw std::runtime_error("the number of repetitions must be >= 1");
#else
    const int countRepetitions = 1;
#endif
#if COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_HIST
    const std::string bwWeightsFile(argv[argIdx++]);
#elif COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_DATAFILE
    const std::string dataFile(argv[argIdx++]);
#endif
    
#ifdef COMPRESSION_DATA_BENCHMARK_RUN
    using varex_t = variant_executor_helper<2, 1, size_t, size_t, size_t, int, unsigned>::type
        ::for_variant_params<std::string, std::string>
        ::for_setting_params<>;
    varex_t varex(
            {"countValuesLarge", "settingGroup", "settingIdx", "repIdx", "bitwidth"},
            {"vector_extension", "format"},
            {}
    );
#endif
    
    MSV_CXX_ATTRIBUTE_PPUNUSED
    const size_t digits = std::numeric_limits<uint64_t>::digits;
    
#if COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_HIST
    // Read the weights of the bit width histograms from a file.
    const uint64_t * bwHists;
    size_t countBwHists;
    std::tie(bwHists, countBwHists) =
            generate_with_bitwidth_histogram_helpers::read_bw_weights(bwWeightsFile);
#elif COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_DISTR
    // Define the data distributions.
    std::vector<std::tuple<size_t, data_generator *>> generators;
    // Exact bit width.
    for(unsigned bw = 1; bw <= digits; bw++)
        generators.push_back(
                {
                    1,
                    new special_data_generator<std::uniform_int_distribution>(
                            std::uniform_int_distribution<uint64_t>(
                                    bitwidth_min<uint64_t>(bw),
                                    bitwidth_max<uint64_t>(bw)
                            )
                    )
                }
        );
    // Uniform distribution.
    for(unsigned bw = 1; bw <= digits; bw++)
        generators.push_back(
                {
                    2,
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
                        3,
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
                            4,
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
    // Amount of outliers.
    for(double outlierShare : {0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}) {
        generators.push_back(
                {
                    5,
                    new special_data_generator<two_value_distribution>(
                            two_value_distribution<uint64_t>(
                                    bitwidth_max<uint64_t>(4),
                                    bitwidth_max<uint64_t>(60),
                                    outlierShare
                            )
                    )
                }
        );
    }
#elif COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_DATAFILE
    // Determine the number of blocks of size countValuesSmall in the data file.
    size_t blockSizeByte = convert_size<uint64_t, uint8_t>(countValuesSmall);
    size_t countBlocksDataFile;
    std::ifstream ifsDataFile(dataFile, std::ios::in | std::ios::binary);
    if(ifsDataFile.good()) {
        ifsDataFile.seekg(0, std::ios_base::end);
        const size_t sizeByteDataFile = ifsDataFile.tellg();
        if(sizeByteDataFile % sizeof(uint64_t))
            throw std::runtime_error(
                    "the size of the data file is not a multiple of 8 byte, "
                    "but we want to read 64 bit integers"
            );
        countBlocksDataFile = sizeByteDataFile / blockSizeByte;
    }
    else
        throw std::runtime_error("could not open the data file for reading");
#endif
    
    
    for(int repIdx = 1; repIdx <= countRepetitions; repIdx++) {
        size_t settingIdx = 0;
        
#if COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_HIST
        for(unsigned bwHistIdx = 0; bwHistIdx < countBwHists; bwHistIdx++) {
            const uint64_t * const bwHist = bwHists + bwHistIdx * digits;
            
            unsigned maxBw = digits;
            while(!bwHist[maxBw - 1])
                maxBw--;
            
            size_t settingGroup = 0;
            
            for(bool isSorted : {false, true}) {
#elif COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_DISTR
        for(auto genInfo : generators) {
            size_t settingGroup;
            data_generator * generator;
            std::tie(settingGroup, generator) = genInfo;
            
            for(bool isSorted : {false, true}) {
#elif COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_DATAFILE
        ifsDataFile.seekg(0, std::ios_base::beg);
        for(size_t blockIdx = 0; blockIdx < countBlocksDataFile; blockIdx++) {
            size_t settingGroup = 0;
            
            { // no explicit sorting if the data is loaded from a file
#endif
            
                settingIdx++;
                std::cerr << "settingIdx: " << settingIdx << std::endl;

#ifdef COMPRESSION_DATA_BENCHMARK_RUN
                varex.print_datagen_started();
#else
                std::cerr << "generating input data column... ";
#endif
#if COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_HIST
                auto origCol = generate_with_bitwidth_histogram(
                        countValuesSmall, bwHist, isSorted, true
                );
#else
    #if COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_DISTR
                // Generate the data.
                auto origCol = generator->generate(countValuesSmall, isSorted);
    #elif COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_DATAFILE
                // Load the next block from the data file.
                auto origCol = new column<uncompr_f>(blockSizeByte);
                ifsDataFile.read(origCol->get_data(), blockSizeByte);
                if(!ifsDataFile.good())
                    throw std::runtime_error(
                            "could not read the next block from the data file"
                    );
                origCol->set_meta_data(countValuesSmall, blockSizeByte);
    #endif
                // Find out the maximum bit width.
                uint64_t pseudoMax = 0;
                const uint64_t * origData = origCol->get_data();
                for(size_t i = 0; i < countValuesSmall; i++)
                    pseudoMax |= origData[i];
                const unsigned maxBw = effective_bitwidth(pseudoMax);
#endif
#ifdef COMPRESSION_DATA_BENCHMARK_RUN
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
#else
                std::cerr << "done.";
            
                MONITORING_CREATE_MONITOR(
                        MONITORING_MAKE_MONITOR(settingGroup, settingIdx),
                        MONITORING_KEY_IDENTS("settingGroup", "settingIdx")
                );

                // Parameters of the data generation.
                MONITORING_ADD_INT_FOR("countValuesSmall", countValuesSmall, settingGroup, settingIdx);

                // The maximum bit widths as used for static_vbp_f.
                MONITORING_ADD_INT_FOR("bitwidth", maxBw, settingGroup, settingIdx);

                // Data characteristics of the (small) input column.
                std::cerr << std::endl << "analyzing input column... ";
                MONITORING_ADD_DATAPROPERTIES_FOR(
                        "", data_properties(origCol), settingGroup, settingIdx
                );
                std::cerr << "done." << std::endl;
#endif

                delete origCol;
            }
        }
    }
     
#if COMPRESSION_DATA_BENCHMARK_DATA_SOURCE == DATA_SOURCE_DISTR
    for(auto genInfo : generators)
        delete std::get<1>(genInfo);
#endif
    
#ifdef COMPRESSION_DATA_BENCHMARK_RUN
    varex.done();
    return !varex.good();
#else
    MONITORING_PRINT_MONITORS(monitorCsvLog);
    return 0;
#endif
}