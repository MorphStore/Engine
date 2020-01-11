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
 * @file const_prof.cpp
 * @brief Measures the constant profiles of logical-level algorithms with a
 * constant behavior required for our cost model for lightweight compression
 * algorithms.
 * 
 * The output is produced as a CSV table on stdout.
 * 
 * @todo Using `variant_executor` here might be a little bit of overkill here
 * and its checks cannot succeed in this setting...
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/delta.h>
#include <core/morphing/for.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/preprocessor.h>
#include <core/utils/variant_executor.h>
#include <vector/vector_extension_names.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vectorlib;


// ****************************************************************************
// Special "format" for doing nothing at the physical level of a cascade.
// ****************************************************************************

struct empty_f : public format {
    static size_t get_size_max_byte(
            MSV_CXX_ATTRIBUTE_PPUNUSED size_t p_CountValues
    ) {
        return 0;
    }

    static const size_t m_BlockSize = 1;
};

template<class t_vector_extension>
struct morph_batch_t<t_vector_extension, empty_f, uncompr_f> {
    static void apply(
            MSV_CXX_ATTRIBUTE_PPUNUSED const uint8_t * & in8,
            MSV_CXX_ATTRIBUTE_PPUNUSED uint8_t * & out8,
            MSV_CXX_ATTRIBUTE_PPUNUSED size_t countLog
    ) {
        // Useless, but we need to do something here, otherwise the compiler
        // optimizes the compression of delta+empty away.
        *out8 = 0x49;
    }
};

template<class t_vector_extension>
struct morph_batch_t<t_vector_extension, uncompr_f, empty_f> {
    static void apply(
            MSV_CXX_ATTRIBUTE_PPUNUSED const uint8_t * & in8,
            MSV_CXX_ATTRIBUTE_PPUNUSED uint8_t * & out8,
            MSV_CXX_ATTRIBUTE_PPUNUSED size_t countLog
    ) {
        // Do nothing.
    }
};


// ****************************************************************************
// Mapping from formats to string names
// ****************************************************************************

// All template-specializations of a format are mapped to a name, which may or
// may not contain the values of the template parameters.
        
template<class t_format>
std::string formatName = "(unknown format)";

template<size_t t_BlockSizeLog, unsigned t_Step>
std::string formatName<delta_f<t_BlockSizeLog, t_Step, empty_f> > =
        "delta_f<" + std::to_string(t_Step) + ">";

template<size_t t_BlockSizeLog, size_t t_PageSizeBlocks>
std::string formatName<for_f<t_BlockSizeLog, t_PageSizeBlocks, empty_f> > =
        "for_f<" + std::to_string(t_PageSizeBlocks) + ">";


// ****************************************************************************
// "Operator" to be executed by `variant_executor`
// ****************************************************************************

/**
 * @brief Measures the compression and decompression time for the specified
 * format.
 * @param p_InCol The column to be compressed and decompressed.
 * @return The decompressed-again input column.
 */
template<class t_vector_extension, class t_format>
const column<uncompr_f> * measure_morphs(const column<uncompr_f> * p_InCol) {
    // This is unused iff monitoring is disabled.
    MSV_CXX_ATTRIBUTE_PPUNUSED const size_t countValues =
        p_InCol->get_count_values();
    
    MONITORING_START_INTERVAL_FOR(
            "runtime compr [µs]",
            veName<t_vector_extension>, formatName<t_format>, countValues
    );
    auto comprCol = morph<t_vector_extension, t_format, uncompr_f>(p_InCol);
    MONITORING_END_INTERVAL_FOR(
            "runtime compr [µs]",
            veName<t_vector_extension>, formatName<t_format>, countValues
    );
            
    MONITORING_START_INTERVAL_FOR(
            "runtime decompr [µs]",
            veName<t_vector_extension>, formatName<t_format>, countValues
    );
    auto decomprCol = morph<t_vector_extension, uncompr_f, t_format>(comprCol);
    MONITORING_END_INTERVAL_FOR(
            "runtime decompr [µs]",
            veName<t_vector_extension>, formatName<t_format>, countValues
    );
    
    if(!std::is_same<t_format, uncompr_f>::value)
        delete comprCol;
    
    return decomprCol;
}


// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(vector_extension, format) { \
    new typename varex_t::operator_wrapper::template for_output_formats<uncompr_f>::template for_input_formats<uncompr_f>( \
        &measure_morphs<vector_extension, format>, true \
    ), \
    veName<vector_extension>, \
    formatName<format>, \
}

#define MAKE_VARIANT_DELTA(ve) \
        MAKE_VARIANT(ve, SINGLE_ARG(delta_f<blockSize, ve::vector_helper_t::element_count::value, empty_f>))

#define MAKE_VARIANT_FOR(ve) \
        MAKE_VARIANT(ve, SINGLE_ARG(for_f<blockSize, ve::vector_helper_t::element_count::value, empty_f>))

#define MAKE_VARIANTS(ve) \
        MAKE_VARIANT_DELTA(ve), \
        MAKE_VARIANT_FOR(ve)


// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 1>::type
        ::for_variant_params<std::string, std::string>
        ::for_setting_params<size_t>;
    varex_t varex(
            {},
            {"vector_extension", "format"},
            {"countValues"}
    );
    
    // @todo This could be a command line argument.
    const size_t countValues = 128 * 1024 * 1024;
    
    const size_t blockSize = 1024;
    
    std::vector<varex_t::variant_t> variants = {
        MAKE_VARIANTS(scalar<v64<uint64_t>>),
#ifdef SSE
        MAKE_VARIANTS(sse<v128<uint64_t>>),
#endif
#ifdef AVXTWO
        MAKE_VARIANTS(avx2<v256<uint64_t>>),
#endif
#ifdef AVX512
        MAKE_VARIANTS(avx512<v512<uint64_t>>),
#endif
    };

    std::cerr << "Failed checks are ok in this microbenchmark" << std::endl;
    varex.print_datagen_started();
    // In fact, the data does not matter here, only the number of data elements
    // is relevant.
    auto origCol = generate_sorted_unique(countValues);
    varex.print_datagen_done();

    varex.execute_variants(variants, countValues, origCol);

    delete origCol;
    
    varex.done();
    std::cerr << "Failed checks are ok in this microbenchmark" << std::endl;
    
    return 0; // Failed checks are ok in this microbenchmark
}