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
 * @file uncompr.cpp
 * @brief Calibration measurements for the uncompressed format.
 * 
 * The output is produced as a CSV table on stdout.
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/operators/general_vectorized/agg_sum_compr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/variant_executor.h>
#include <vector/vector_extension_names.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <stdexcept>
#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vectorlib;


// ****************************************************************************
// "Operator" to be executed by `variant_executor`
// ****************************************************************************

template<class t_vector_extension>
const column<uncompr_f> * reg2ram(size_t p_CountRam) {
    using t_ve = t_vector_extension;
    IMPORT_VECTOR_BOILER_PLATE(t_ve)
            
    if(p_CountRam % vector_element_count::value)
        throw std::runtime_error(
                "the number of data elements to store to RAM must be a "
                "multiple of the number of data elements per vector register"
        );
    
    const size_t size = get_size_max_byte_any_len<uncompr_f>(p_CountRam);
    auto outCol = new column<uncompr_f>(size);
    uint64_t * outData = outCol->get_data();
    
    const vector_t vec = set1<t_ve, vector_base_t_granularity::value>(123);
    for(size_t i = 0; i < p_CountRam; i += vector_element_count::value)
        store<t_ve, iov::ALIGNED, vector_size_bit::value>(outData + i, vec);
    
    outCol->set_meta_data(p_CountRam, size);
    return outCol;
}

template<class t_vector_extension>
const column<uncompr_f> * cache2ram(
        const column<uncompr_f> * p_InCol,
        size_t p_CountCache,
        size_t p_CountRam
) {
    using t_ve = t_vector_extension;
    IMPORT_VECTOR_BOILER_PLATE(t_ve)
            
    if(p_CountCache % vector_element_count::value)
        throw std::runtime_error(
                "the number of data elements to load from cache must be a "
                "multiple of the number of data elements per vector register"
        );
    if(p_CountRam % p_CountCache)
        throw std::runtime_error(
                "the number of data elements to store to RAM must be a "
                "multiple of the number of data elements to load from cache"
        );
            
    const uint64_t * inData = p_InCol->get_data();
            
    const size_t size = get_size_max_byte_any_len<uncompr_f>(p_CountRam);
    auto outCol = new column<uncompr_f>(size);
    uint64_t * outData = outCol->get_data();
    
    const size_t countBlocks = p_CountRam / p_CountCache;
    for(size_t blockIdx = 0; blockIdx < countBlocks; blockIdx++)
        for(size_t i = 0; i < p_CountCache; i += vector_element_count::value) {
            store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                    outData,
                    load<t_ve, iov::ALIGNED, vector_size_bit::value>(
                            inData + i
                    )
            );
            outData += vector_element_count::value;
        }
    
    outCol->set_meta_data(p_CountRam, size);
    return outCol;
}

template<class t_vector_extension>
const column<uncompr_f> * measure(
        const column<uncompr_f> * p_InCol, int p_RepIdx
) {
    const size_t countValues = p_InCol->get_count_values();
    
    MONITORING_START_INTERVAL_FOR(
            "runtime agg [µs]",
            veName<t_vector_extension>, p_RepIdx
    );
    auto sumCol = agg_sum<t_vector_extension>(p_InCol);
    MONITORING_END_INTERVAL_FOR(
            "runtime agg [µs]",
            veName<t_vector_extension>, p_RepIdx
    );
    
    MONITORING_START_INTERVAL_FOR(
            "runtime reg2ram [µs]",
            veName<t_vector_extension>, p_RepIdx
    );
    auto uselessColReg2Ram = reg2ram<t_vector_extension>(countValues);
    MONITORING_END_INTERVAL_FOR(
            "runtime reg2ram [µs]",
            veName<t_vector_extension>, p_RepIdx
    );
    delete uselessColReg2Ram;
    
    MONITORING_START_INTERVAL_FOR(
            "runtime cache2ram [µs]",
            veName<t_vector_extension>, p_RepIdx
    );
    auto uselessColCache2Ram = cache2ram<t_vector_extension>(
            p_InCol,
            write_iterator_base<t_vector_extension, uncompr_f>::m_CountBuffer,
            countValues
    );
    MONITORING_END_INTERVAL_FOR(
            "runtime cache2ram [µs]",
            veName<t_vector_extension>, p_RepIdx
    );
    delete uselessColCache2Ram;
    
    return sumCol;
}


// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(vector_extension) { \
    new typename varex_t::operator_wrapper::template for_output_formats<uncompr_f>::template for_input_formats<uncompr_f>( \
        &measure<vector_extension> \
    ), \
    veName<vector_extension> \
}

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(int argc, char ** argv) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    if(argc != 2)
        throw std::runtime_error(
                "this calibration benchmark expect the number of repetitions "
                "as its only argument"
        );
    const int countRepetitions = atoi(argv[1]);
    if(countRepetitions < 1)
        throw std::runtime_error("the number of repetitions must be >= 1");
    
    using varex_t = variant_executor_helper<1, 1, int>::type
        ::for_variant_params<std::string>
        ::for_setting_params<>;
    varex_t varex(
            {"repetition"},
            {"vector_extension"},
            {}
    );
    
    // @todo This could be a command line argument.
    const size_t countValues = 128 * 1024 * 1024;
    
    std::vector<varex_t::variant_t> variants = {
        MAKE_VARIANT(scalar<v64<uint64_t>>),
#ifdef SSE
        MAKE_VARIANT(sse<v128<uint64_t>>),
#endif
#ifdef AVXTWO
        MAKE_VARIANT(avx2<v256<uint64_t>>),
#endif
#ifdef AVX512
        MAKE_VARIANT(avx512<v512<uint64_t>>),
#endif
    };

    varex.print_datagen_started();
    // In fact, the data does not matter here, only the number of data elements
    // is relevant.
    auto origCol = generate_sorted_unique(countValues);
    varex.print_datagen_done();

    for(int repIdx = 1; repIdx <= countRepetitions; repIdx++)
        varex.execute_variants(variants, origCol, repIdx);

    delete origCol;
    
    varex.done();
    
    return !varex.good();
}