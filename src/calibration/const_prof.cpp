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
#include <core/operators/general_vectorized/agg_sum_compr.h>
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

#include <cstdlib>

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
        // Useless, but we need to store something to the output buffer,
        // otherwise the compiler may have problems compiling delta_f with
        // empty_f, since it assumes the cascade's internal buffer to be
        // uninitialized.
        *out8 = 0x49;
    }
};


// ****************************************************************************
// Mapping from formats to string names
// ****************************************************************************

// These name mappings are so specific to this micro benchmark that they are
// not included in <core/morphing/format_names.h>, but the mechanism is the
// same.

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
// Variant of the morph-operator simulating the use on the output side of
// on-the-fly de/re-compression at the block-granularity.
// ****************************************************************************

// ----------------------------------------------------------------------------
// Interface
// ----------------------------------------------------------------------------

/**
 * @brief A variant of the morph-operator simulating the output side of
 * on-the-fly de/re-compression at the block-granularity, where data is read
 * from the internal buffer of a write-iterator and written to the internal
 * buffer of a cascade, for logical-level algorithms.
 * 
 * That is, data must effectively be read from cache and written to cache.
 */
template<class t_vector_extension, class t_dst_f, class t_src_f>
struct cache2cache_morph_t {
    /**
     * 
     * @param p_InCol
     * @param p_BufferCountValues The number of data elements in the internal
     * buffer of the assumed write-iterator.
     * @return 
     */
    static const column<t_dst_f> * apply(
            const column<t_src_f> * p_InCol, size_t p_BufferCountValues
    ) = delete;
};

// ----------------------------------------------------------------------------
// Compression
// ----------------------------------------------------------------------------

template<class t_vector_extension, class t_dst_f>
struct cache2cache_morph_t<t_vector_extension, t_dst_f, uncompr_f> {
    static const column<t_dst_f> * apply(
            const column<uncompr_f> * p_InCol, size_t p_BufferCountValues
    ) {
        const size_t srcCountValues = p_InCol->get_count_values();
        
        if(srcCountValues % p_BufferCountValues)
            throw std::runtime_error(
                    "cache2cache_morph_t compression: the number of data "
                    "elements in the input column must be a multiple of the "
                    "assumed number of data elements in the buffer"
            );
        if(p_BufferCountValues % t_dst_f::m_BlockSize)
            throw std::runtime_error(
                    "cache2cache_morph_t compression: the assumed number of "
                    "data elements in the buffer must be a multiple of the "
                    "destination format's block size"
            );
        
        auto outCol = new column<t_dst_f>(
                t_dst_f::get_size_max_byte(p_BufferCountValues)
        );
        uint8_t * outData = outCol->get_data();
        uint8_t * const initOutData = outData;
        
        // Both, the input pointer and the output pointer are re-used for all
        // blocks to achieve reading from and writing to the cache all the time.
        size_t countCompressed = 0;
        while(countCompressed < srcCountValues) {
            const uint8_t * inData = p_InCol->get_data();
            outData = initOutData;
            morph_batch<t_vector_extension, t_dst_f, uncompr_f>(
                    inData, outData, p_BufferCountValues
            );
            countCompressed += p_BufferCountValues;
        }
        
        outCol->set_meta_data(p_BufferCountValues, outData - initOutData);
        return outCol;
    }
};


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
const column<uncompr_f> * measure_morphs(
        const column<uncompr_f> * p_InCol, int p_RepIdx
) {
    // This is unused iff monitoring is disabled.
    MSV_CXX_ATTRIBUTE_PPUNUSED const size_t countValues =
        p_InCol->get_count_values();
    
    MONITORING_START_INTERVAL_FOR(
            "runtime compr [µs]",
            veName<t_vector_extension>, formatName<t_format>, countValues, p_RepIdx
    );
    auto comprCol = morph<t_vector_extension, t_format, uncompr_f>(p_InCol);
    MONITORING_END_INTERVAL_FOR(
            "runtime compr [µs]",
            veName<t_vector_extension>, formatName<t_format>, countValues, p_RepIdx
    );
            
    MONITORING_START_INTERVAL_FOR(
            "runtime decompr [µs]",
            veName<t_vector_extension>, formatName<t_format>, countValues, p_RepIdx
    );
    auto decomprCol = morph<t_vector_extension, uncompr_f, t_format>(comprCol);
    MONITORING_END_INTERVAL_FOR(
            "runtime decompr [µs]",
            veName<t_vector_extension>, formatName<t_format>, countValues, p_RepIdx
    );
            
    MONITORING_START_INTERVAL_FOR(
            "runtime agg [µs]",
            veName<t_vector_extension>, formatName<t_format>, countValues, p_RepIdx
    );
    auto sumCol = agg_sum<t_vector_extension, t_format>(comprCol);
    MONITORING_END_INTERVAL_FOR(
            "runtime agg [µs]",
            veName<t_vector_extension>, formatName<t_format>, countValues, p_RepIdx
    );
            
    MONITORING_START_INTERVAL_FOR(
            "runtime compr cache2cache [µs]",
            veName<t_vector_extension>, formatName<t_format>, countValues, p_RepIdx
    );
    auto comprColCache2Cache = cache2cache_morph_t<
            t_vector_extension, t_format, uncompr_f
    >::apply(
            p_InCol,
            write_iterator_base<t_vector_extension, t_format>::m_CountBuffer
    );
    MONITORING_END_INTERVAL_FOR(
            "runtime compr cache2cache [µs]",
            veName<t_vector_extension>, formatName<t_format>, countValues, p_RepIdx
    );
    
    if(!std::is_same<t_format, uncompr_f>::value)
        delete comprCol;
    delete sumCol;
    delete comprColCache2Cache;
    
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
    using varex_t = variant_executor_helper<1, 1, int>::type
        ::for_variant_params<std::string, std::string>
        ::for_setting_params<size_t>;
    varex_t varex(
            {"repetition"},
            {"vector_extension", "format"},
            {"countValues"}
    );
    
    // @todo This could be a command line argument.
    const size_t countValues = 128 * 1024 * 1024;
    
    const size_t blockSize = 1024;
    
    std::vector<varex_t::variant_t> variants = {
#ifdef AVX512
        MAKE_VARIANTS(avx512<v512<uint64_t>>),
#elif defined(AVXTWO)
        MAKE_VARIANTS(avx2<v256<uint64_t>>),
#elif defined(SSE)
        MAKE_VARIANTS(sse<v128<uint64_t>>),
#else
        MAKE_VARIANTS(scalar<v64<uint64_t>>),
#endif
    };

    std::cerr << "Failed checks are ok in this microbenchmark" << std::endl;
    varex.print_datagen_started();
    // In fact, the data does not matter here, only the number of data elements
    // is relevant.
    auto origCol = generate_sorted_unique(countValues);
    varex.print_datagen_done();

    for(int repIdx = 1; repIdx <= countRepetitions; repIdx++)
        varex.execute_variants(variants, countValues, origCol, repIdx);

    delete origCol;
    
    varex.done();
    std::cerr << "Failed checks are ok in this microbenchmark" << std::endl;
    
    return 0; // Failed checks are ok in this microbenchmark
}