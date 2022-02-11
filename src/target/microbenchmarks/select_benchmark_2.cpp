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
 * @file select_benchmark_2.cpp
 * @brief Another micro benchmark of the select operator.
 * 
 * If the macro SELECT_BENCHMARK_2_TIME is defined, then the code will do time
 * measurements of different variants of the select-operator. Otherwise, it
 * will record the data characteristics of the input and output data. This is
 * configured in the CMakeLists-file.
 */

//#define SELECT_BENCHMARK_2_TIME

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <vector/vector_extension_names.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#ifdef SELECT_BENCHMARK_2_TIME
#include <core/morphing/default_formats.h>
#include <core/morphing/delta.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/for.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/vbp.h>
#include <core/morphing/format_names.h> // Must be included after all formats.
#include <core/operators/general_vectorized/agg_sum_compr.h>
#include <core/operators/general_vectorized/select_compr.h>
#include <core/utils/variant_executor.h>
#else
#include <core/operators/scalar/select_uncompr.h>
#include <core/utils/data_properties.h>
#include <core/utils/monitoring.h>
#endif

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include <cstring>

using namespace morphstore;
using namespace vectorlib;


#ifdef SELECT_BENCHMARK_2_TIME

// ****************************************************************************
// A modified morph-operator.
// ****************************************************************************
// When morphing from uncompressed to uncompressed, it does not merely return
// the input pointer, as the normal morph-operator, but does a memcpy. We need
// this behavior for this micro-benchmark.

template<class t_vector_extension, class t_dst_f, class t_src_f>
struct morph_or_copy_t {
    static const column<t_dst_f> * apply(const column<t_src_f> * p_InCol) {
        return morph<t_vector_extension, t_dst_f>(p_InCol);
    }
};

template<class t_vector_extension>
struct morph_or_copy_t<t_vector_extension, uncompr_f, uncompr_f> {
    static const column<uncompr_f> * apply(const column<uncompr_f> * p_InCol) {
        const size_t sizeByte = p_InCol->get_size_used_byte();
        auto outCol = new column<uncompr_f>(sizeByte);
        memcpy(outCol->get_data(), p_InCol->get_data(), sizeByte);
        outCol->set_meta_data(p_InCol->get_count_values(), sizeByte);
        return outCol;
    }
};


// ****************************************************************************
// Wrapper of the select-operator to be executed by `variant_executor`
// ****************************************************************************

template<class t_vector_extension, class t_out_pos_f, class t_in_data_f>
const column<t_out_pos_f> * measure_select_and_morphs(
        const column<t_in_data_f> * p_InDataColCompr,
        uint64_t p_Pred,
        // Unused iff monitoring is disabled.
        MSV_CXX_ATTRIBUTE_PPUNUSED unsigned p_DatasetIdx
) {
    // We go from compressed inDataCol to compressed outPosCol via two ways to
    // measure both the actual select-operator and the morphs involved in it.
    // Furthermore, we do an aggregation on the compressed input column to
    // measure the time for decompression without the materialization of the
    // uncompressed data.
    
    
    // 1) Select-operator on compressed data.
    
    MONITORING_START_INTERVAL_FOR(
            "runtime select:µs",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    auto outPosColCompr = my_select_wit_t<
            equal, t_vector_extension, t_out_pos_f, t_in_data_f
    >::apply(p_InDataColCompr, p_Pred);
    MONITORING_END_INTERVAL_FOR(
            "runtime select:µs",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    
    
#if 0
    // 2) Decompression, select-operator on uncompressed data, recompression.
    
    MONITORING_START_INTERVAL_FOR(
            "runtime decompr:µs",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    auto inDataColUncompr = morph_or_copy_t<
            t_vector_extension, uncompr_f, t_in_data_f
    >::apply(p_InDataColCompr);
    MONITORING_END_INTERVAL_FOR(
            "runtime decompr:µs",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    
    auto outPosColUncompr = my_select_wit_t<
            equal, t_vector_extension, uncompr_f, uncompr_f
    >::apply(inDataColUncompr, p_Pred);

    // This condition would be necessary with the normal morph-operator.
    // if(!std::is_same<t_in_data_f, uncompr_f>::value)
    delete inDataColUncompr;
    
    MONITORING_START_INTERVAL_FOR(
            "runtime recompr:µs",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    auto outPosColRecompr = morph_or_copy_t<
            t_vector_extension, t_out_pos_f, uncompr_f
    >::apply(outPosColUncompr);
    MONITORING_END_INTERVAL_FOR(
            "runtime recompr:µs",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    
    // This condition would be necessary with the normal morph-operator.
    // if(!std::is_same<t_out_pos_f, uncompr_f>::value)
    delete outPosColUncompr;
    
    MONITORING_ADD_BOOL_FOR(
            "count-check detour",
            outPosColCompr->get_count_values() == outPosColRecompr->get_count_values(),
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    
    delete outPosColRecompr;
    
    
    // 3) Aggregation.
    
    MONITORING_START_INTERVAL_FOR(
            "runtime agg_sum:µs",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    auto sumCol = agg_sum<t_vector_extension>(p_InDataColCompr);
    MONITORING_END_INTERVAL_FOR(
            "runtime agg_sum:µs",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    
    // Record the sum to prevent the compiler from optimizing it away.
    MONITORING_ADD_INT_FOR(
            "sumInDataCol",
            *static_cast<const uint64_t *>(sumCol->get_data()),
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    
    delete sumCol;
#endif
    
    
    return outPosColCompr;
}

// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(ve, out_pos_f, in_data_f) { \
    new typename t_varex_t::operator_wrapper::template for_output_formats<out_pos_f>::template for_input_formats<in_data_f>( \
        &measure_select_and_morphs<ve, out_pos_f, in_data_f> \
    ), \
    veName<ve>, \
    formatName<out_pos_f>, \
    formatName<in_data_f> \
}

#if 1
#define MAKE_VARIANTS_VE_OUT(ve, out_pos_f, inBw) \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_f), \
            uncompr_f \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_f), \
            DEFAULT_STATIC_VBP_F(ve, inBw) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_f), \
            DEFAULT_DYNAMIC_VBP_F(ve) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_f), \
            DEFAULT_DELTA_DYNAMIC_VBP_F(ve) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_f), \
            DEFAULT_FOR_DYNAMIC_VBP_F(ve) \
    )

#define MAKE_VARIANTS_VE(ve, outBw, inBw) \
    MAKE_VARIANTS_VE_OUT( \
            ve, \
            uncompr_f, \
            inBw \
    ), \
    MAKE_VARIANTS_VE_OUT( \
            ve, \
            DEFAULT_STATIC_VBP_F(ve, outBw), \
            inBw \
    ), \
    MAKE_VARIANTS_VE_OUT( \
            ve, \
            DEFAULT_DYNAMIC_VBP_F(ve), \
            inBw \
    ), \
    MAKE_VARIANTS_VE_OUT( \
            ve, \
            DEFAULT_DELTA_DYNAMIC_VBP_F(ve), \
            inBw \
    ), \
    MAKE_VARIANTS_VE_OUT( \
            ve, \
            DEFAULT_FOR_DYNAMIC_VBP_F(ve), \
            inBw \
    )
#else
#define MAKE_VARIANTS_VE_OUT(ve, out_pos_f, inBw) \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_f), \
            uncompr_f \
    )

#define MAKE_VARIANTS_VE(ve, outBw, inBw) \
    MAKE_VARIANTS_VE_OUT( \
            ve, \
            uncompr_f, \
            inBw \
    )
#endif

template<class t_varex_t, unsigned t_OutBw, unsigned t_InBw>
std::vector<typename t_varex_t::variant_t> make_variants() {
    return {
#ifdef AVX512
        MAKE_VARIANTS_VE(avx512<v512<uint64_t>>, t_OutBw, t_InBw),
#elif defined(AVXTWO)
        MAKE_VARIANTS_VE(avx2<v256<uint64_t>>, t_OutBw, t_InBw),
#elif defined(SSE)
        MAKE_VARIANTS_VE(sse<v128<uint64_t>>, t_OutBw, t_InBw),
#else
        MAKE_VARIANTS_VE(scalar<v64<uint64_t>>, t_OutBw, t_InBw),
#endif
    };
}

#define VG_BEGIN \
    if(false) {/*dummy*/}
// @todo This should be effective_bitwidth(_inDataCount - 1).
#define VG_CASE(_inDataCount, _inDataMax) \
    else if(inDataCount == _inDataCount && inDataMax == _inDataMax) \
        variants = make_variants< \
                varex_t, \
                effective_bitwidth(_inDataCount), \
                effective_bitwidth(_inDataMax) \
        >();
#define VG_END \
    else throw std::runtime_error( \
            "unexpected combination: inDataCount=" + \
            std::to_string(inDataCount) + ", inDataMax=" + \
            std::to_string(inDataMax) \
    );

#endif

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    const size_t inDataCount = 128 * 1024 * 1024;
    
#ifdef SELECT_BENCHMARK_2_TIME
    // The datasetIdx is actually a setting parameter, but we need to model it
    // as an additional parameter to be able to hand it into our variant
    // function measure_select_and_morphs.
    using varex_t = variant_executor_helper<1, 1, uint64_t, unsigned>::type
        ::for_variant_params<std::string, std::string, std::string>
        ::for_setting_params<>;
    varex_t varex(
            {"pred", "datasetIdx"},
            {"vector_extension", "out_pos_f", "in_data_f"},
            {}
    );
    
#endif
    
    // @todo It would be nice to use a 64-bit value, but then, some vector-lib
    // primitives would interpret it as a negative number. This would hurt,
    // e.g., FOR.
    // Looks strange, but saves us from casting in the initializer list.
    const uint64_t _0 = 0;
    const uint64_t _63 = 63;
    const uint64_t _100k = 100000;
    const uint64_t min48bit = bitwidth_min<uint64_t>(48);
    const uint64_t min63bit = bitwidth_min<uint64_t>(63);
    const uint64_t max63bit = bitwidth_max<uint64_t>(63);
    
    unsigned datasetIdx = 0;
    for(float selectedShare : {
        0.01,
        0.9,
    }) {
        bool isSorted;
        uint64_t mainMin;
        uint64_t mainMax;
        uint64_t outlierMin;
        uint64_t outlierMax;
        double outlierShare;
        MSV_CXX_ATTRIBUTE_PPUNUSED bool inDataAssumedUnique;
        
        for(auto params : {
            // Cx (Cy): number in PD's PhD thesis (number in the PVLDB-paper).
            
            // Unsorted, small numbers, no outliers -> good for static_vbp.
            std::make_tuple(false, _0, _63, _0, _0, 0.0, false), // C2 (C1)
            // Unsorted, small numbers, very rare outliers -> good for dynamic_vbp.
            std::make_tuple(false, _0, _63, max63bit, max63bit, 0.0001, false), // C3 (C2)
            // Unsorted, huge numbers in narrow range, no outliers -> good for for+dynamic_vbp.
            std::make_tuple(false, min63bit, min63bit + 63, _0, _0, 0.0, false), // C4 (C3)
            // Sorted, large numbers -> good for delta+dynamic_vbp.
            std::make_tuple(true, _0, _100k, _0, _0, 0.0, false), // C5 (not used)
            // Sorted, large numbers -> good for delta+dynamic_vbp.
            std::make_tuple(true, min48bit, min48bit + _100k, _0, _0, 0.0, false), // C6 (C4)
            // Unsorted, random numbers -> good for nothing/uncompr.
            std::make_tuple(false, _0, max63bit, _0, _0, 0.0, true), // C7 (C5)
        }) {
            datasetIdx++;
            
            std::tie(
                    isSorted,
                    mainMin, mainMax,
                    outlierMin, outlierMax,
                    outlierShare,
                    inDataAssumedUnique
            ) = params;
            MSV_CXX_ATTRIBUTE_PPUNUSED const uint64_t inDataMax = \
                    std::max(mainMax, outlierMax);

#ifdef SELECT_BENCHMARK_2_TIME
            varex.print_datagen_started();
#else
            std::cerr << "generating input data column... ";
#endif
            auto inDataCol = ColumnGenerator::generate_with_outliers_and_selectivity(
                    inDataCount,
                    mainMin, mainMax,
                    selectedShare,
                    outlierMin, outlierMax, outlierShare,
                    isSorted
            );
#ifdef SELECT_BENCHMARK_2_TIME
            varex.print_datagen_done();
            
            std::vector<varex_t::variant_t> variants;
        
            // Only enumerate the maximum bit widths that might actually be
            // encountered depending on the parameters of the data generation.
            // We do not need all 64 bit widths. This greatly reduces the
            // compilation time.
            VG_BEGIN
            VG_CASE(inDataCount, _63)
            VG_CASE(inDataCount, max63bit)
            VG_CASE(inDataCount, min63bit + 63)
            VG_CASE(inDataCount, _100k)
            VG_CASE(inDataCount, min48bit + _100k)
            VG_CASE(inDataCount, bitwidth_min<uint64_t>(63))
            VG_END
            
            varex.execute_variants(variants, inDataCol, mainMin, datasetIdx);
#else
            std::cerr << "done.";
            
            MONITORING_CREATE_MONITOR(
                    MONITORING_MAKE_MONITOR(datasetIdx),
                    MONITORING_KEY_IDENTS("datasetIdx")
            );
            
            // Parameters of the data generation.
            MONITORING_ADD_DOUBLE_FOR(
                    "param_selectedShare", selectedShare, datasetIdx
            );
            MONITORING_ADD_DOUBLE_FOR(
                    "param_outlierShare", outlierShare, datasetIdx
            );
            MONITORING_ADD_INT_FOR("param_mainMin", mainMin, datasetIdx);
            MONITORING_ADD_INT_FOR("param_mainMax", mainMax, datasetIdx);
            MONITORING_ADD_INT_FOR("param_outlierMin", outlierMin, datasetIdx);
            MONITORING_ADD_INT_FOR("param_outlierMax", outlierMax, datasetIdx);
            
            // The maximum bit widths as used for static_vbp_f.
            MONITORING_ADD_INT_FOR("inMaxBw", effective_bitwidth(inDataMax), datasetIdx);
            // @todo This should be effective_bitwidth(inDataCount - 1).
            MONITORING_ADD_INT_FOR("outMaxBw", effective_bitwidth(inDataCount), datasetIdx);
            
            // Data characteristics of the input data column.
            std::cerr << std::endl << "analyzing input data column... ";
            MONITORING_ADD_INT_FOR(
                    "inData_ValueCount",
                    inDataCol->get_count_values(),
                    datasetIdx
            );
            MONITORING_ADD_DATAPROPERTIES_FOR(
                    "inData_", data_properties(inDataCol, inDataAssumedUnique), datasetIdx
            );
            std::cerr << "done." << std::endl;
            
            // Execution of the (wrapper of the) select-operator.
            std::cerr << "executing select-operator... ";
            auto outPosCol = select_t<
                    std::equal_to, scalar<v64<uint64_t>>, uncompr_f, uncompr_f
            >::apply(inDataCol, mainMin);
            std::cerr << "done." << std::endl;
            
            // Data characteristics of the output positions column.
            std::cerr << "analyzing output positions column... ";
            MONITORING_ADD_INT_FOR(
                    "outPos_ValueCount",
                    outPosCol->get_count_values(),
                    datasetIdx
            );
            MONITORING_ADD_DATAPROPERTIES_FOR(
                    "outPos_", data_properties(outPosCol, true), datasetIdx
            );
            std::cerr << "done." << std::endl << std::endl;
            
            delete outPosCol;
#endif

            delete inDataCol;
        }
    }
    
#ifdef SELECT_BENCHMARK_2_TIME
    varex.done();
#else
    MONITORING_PRINT_MONITORS(monitorCsvLog);
#endif
    
    return !varex.good();
}
