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

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#ifdef SELECT_BENCHMARK_2_TIME
#include <core/morphing/delta.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/for.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/vbp.h>
#include <core/operators/general_vectorized/select_compr.h>
#include <core/utils/variant_executor.h>
#else
#include <core/operators/scalar/select_uncompr.h>
#include <core/utils/data_properties.h>
#include <core/utils/monitoring.h>
#endif

#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vectorlib;


#ifdef SELECT_BENCHMARK_2_TIME

// ****************************************************************************
// Mapping from vector extensions and formats to string names
// ****************************************************************************
// @todo The same thing exists in the calibration benchmark. Reduce the code
// duplication.

// ----------------------------------------------------------------------------
// Vector extensions
// ----------------------------------------------------------------------------

template<class t_vector_extension>
std::string veName = "(unknown vector extension)";

#define MAKE_VECTOR_EXTENSION_NAME(ve) \
    template<> std::string veName<ve> = STR_EVAL_MACROS(ve);

MAKE_VECTOR_EXTENSION_NAME(scalar<v64<uint64_t>>)
#ifdef SSE
MAKE_VECTOR_EXTENSION_NAME(sse<v128<uint64_t>>)
#endif
#ifdef AVXTWO
MAKE_VECTOR_EXTENSION_NAME(avx2<v256<uint64_t>>)
#endif
#ifdef AVX512
MAKE_VECTOR_EXTENSION_NAME(avx512<v512<uint64_t>>)
#endif

// ----------------------------------------------------------------------------
// Formats
// ----------------------------------------------------------------------------
// All template-specializations of a format are mapped to a name, which may or
// may not contain the values of the template parameters.

template<class t_format>
std::string formatName = "(unknown format)";

template<size_t t_BlockSizeLog, size_t t_PageSizeBlocks, unsigned t_Step>
std::string formatName<
        dynamic_vbp_f<t_BlockSizeLog, t_PageSizeBlocks, t_Step>
> = "dynamic_vbp_f<" + std::to_string(t_BlockSizeLog) + ", " + std::to_string(t_PageSizeBlocks) + ", " + std::to_string(t_Step) + ">";

//template<size_t t_BlockSizeLog>
//std::string formatName<k_wise_ns_f<t_BlockSizeLog>> = "k_wise_ns_f<" + std::to_string(t_BlockSizeLog) + ">";

template<unsigned t_Bw, unsigned t_Step>
std::string formatName<
        static_vbp_f<vbp_l<t_Bw, t_Step> >
> = "static_vbp_f<vbp_l<bw, " + std::to_string(t_Step) + "> >";

template<size_t t_BlockSizeLog, unsigned t_Step, class t_inner_f>
std::string formatName<
        delta_f<t_BlockSizeLog, t_Step, t_inner_f>
> = "delta_f<" + std::to_string(t_BlockSizeLog) + ", " + std::to_string(t_Step) + ", " + formatName<t_inner_f> + ">";

template<size_t t_BlockSizeLog, size_t t_PageSizeBlocks, class t_inner_f>
std::string formatName<
        for_f<t_BlockSizeLog, t_PageSizeBlocks, t_inner_f>
> = "for_f<" + std::to_string(t_BlockSizeLog) + ", " + std::to_string(t_PageSizeBlocks) + ", " + formatName<t_inner_f> + ">";

template<>
std::string formatName<uncompr_f> = "uncompr_f";


// ****************************************************************************
// Macros for the formats.
// ****************************************************************************

#define STATIC_VBP_FORMAT(ve, bw) \
    SINGLE_ARG(static_vbp_f<vbp_l<bw, ve::vector_helper_t::element_count::value>>)

#define DYNAMIC_VBP_FORMAT(ve) \
    SINGLE_ARG(dynamic_vbp_f< \
            ve::vector_helper_t::size_bit::value, \
            ve::vector_helper_t::size_byte::value, \
            ve::vector_helper_t::element_count::value \
    >)

#define DELTA_DYNAMIC_VBP_FORMAT(ve) \
    SINGLE_ARG(delta_f< \
            1024, \
            ve::vector_helper_t::element_count::value, \
            dynamic_vbp_f< \
                    ve::vector_helper_t::size_bit::value, \
                    ve::vector_helper_t::size_byte::value, \
                    ve::vector_helper_t::element_count::value \
            > \
    >)

#define FOR_DYNAMIC_VBP_FORMAT(ve) \
    SINGLE_ARG(for_f< \
            1024, \
            ve::vector_helper_t::element_count::value, \
            dynamic_vbp_f< \
                    ve::vector_helper_t::size_bit::value, \
                    ve::vector_helper_t::size_byte::value, \
                    ve::vector_helper_t::element_count::value \
            > \
    >)


// ****************************************************************************
// Wrapper of the select-operator to be executed by `variant_executor`
// ****************************************************************************

template<class t_vector_extension, class t_out_pos_f, class t_in_data_f>
const column<t_out_pos_f> * measure_select_and_morphs(
        const column<t_in_data_f> * p_InDataColCompr,
        uint64_t p_Pred,
        unsigned p_DatasetIdx
) {
    // We go from compressed inDataCol to compressed outPosCol via two ways to
    // measure both the actual select-operator and the morphs involved in it.
    
    
    // 1) Select-operator on compressed data.
    
    MONITORING_START_INTERVAL_FOR(
            "runtime select [µs]",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    auto outPosColCompr = my_select_wit_t<
            equal, t_vector_extension, t_out_pos_f, t_in_data_f
    >::apply(p_InDataColCompr, p_Pred);
    MONITORING_END_INTERVAL_FOR(
            "runtime select [µs]",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    
    
    // 2) Decompression, select-operator on uncompressed data, recompression.
    
    MONITORING_START_INTERVAL_FOR(
            "runtime decompr [µs]",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    auto inDataColUncompr =
            morph<t_vector_extension, uncompr_f, t_in_data_f
    >(p_InDataColCompr);
    MONITORING_END_INTERVAL_FOR(
            "runtime decompr [µs]",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    
    auto outPosColUncompr = my_select_wit_t<
            equal, t_vector_extension, uncompr_f, uncompr_f
    >::apply(inDataColUncompr, p_Pred);
    
    if(!std::is_same<t_in_data_f, uncompr_f>::value)
        delete inDataColUncompr;
    
    MONITORING_START_INTERVAL_FOR(
            "runtime recompr [µs]",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    auto outPosColRecompr =
            morph<t_vector_extension, t_out_pos_f, uncompr_f>(outPosColUncompr);
    MONITORING_END_INTERVAL_FOR(
            "runtime recompr [µs]",
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    
    if(!std::is_same<t_out_pos_f, uncompr_f>::value)
        delete outPosColUncompr;
    
    MONITORING_ADD_BOOL_FOR(
            "count-check detour",
            outPosColCompr->get_count_values() == outPosColRecompr->get_count_values(),
            veName<t_vector_extension>, formatName<t_out_pos_f>,
            formatName<t_in_data_f>, p_Pred, p_DatasetIdx
    );
    
    delete outPosColRecompr;
    
    
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
            STATIC_VBP_FORMAT(ve, inBw) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_f), \
            DYNAMIC_VBP_FORMAT(ve) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_f), \
            DELTA_DYNAMIC_VBP_FORMAT(ve) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_f), \
            FOR_DYNAMIC_VBP_FORMAT(ve) \
    )

#define MAKE_VARIANTS_VE(ve, outBw, inBw) \
    MAKE_VARIANTS_VE_OUT( \
            ve, \
            uncompr_f, \
            inBw \
    ), \
    MAKE_VARIANTS_VE_OUT( \
            ve, \
            STATIC_VBP_FORMAT(ve, outBw), \
            inBw \
    ), \
    MAKE_VARIANTS_VE_OUT( \
            ve, \
            DYNAMIC_VBP_FORMAT(ve), \
            inBw \
    ), \
    MAKE_VARIANTS_VE_OUT( \
            ve, \
            DELTA_DYNAMIC_VBP_FORMAT(ve), \
            inBw \
    ), \
    MAKE_VARIANTS_VE_OUT( \
            ve, \
            FOR_DYNAMIC_VBP_FORMAT(ve), \
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
        // Compressed variants.
        MAKE_VARIANTS_VE(scalar<v64<uint64_t>>, t_OutBw, t_InBw),
#ifdef SSE
        MAKE_VARIANTS_VE(sse<v128<uint64_t>>, t_OutBw, t_InBw),
#endif
#ifdef AVXTWO
        MAKE_VARIANTS_VE(avx2<v256<uint64_t>>, t_OutBw, t_InBw),
#endif
#ifdef AVX512
        MAKE_VARIANTS_VE(avx512<v512<uint64_t>>, t_OutBw, t_InBw),
#endif
    };
}

#endif

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    const size_t countValues = 128 * 1024 * 1024;
    const unsigned outMaxBw = effective_bitwidth(countValues - 1);
    
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
    const uint64_t largeVal = bitwidth_max<uint64_t>(63);
    // Looks strange, but saves us from casting in the initializer list.
    const uint64_t _0 = 0;
    const uint64_t _63 = 63;
    
    unsigned datasetIdx = 0;
    for(float selectedShare : {
        0.00001,
        0.0001,
        0.001,
        0.01,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9
    }) {
        bool isSorted;
        uint64_t mainMin;
        uint64_t mainMax;
        uint64_t outlierMin;
        uint64_t outlierMax;
        double outlierShare;
        
        for(auto params : {
            // Unsorted, small numbers, no outliers -> good for static_vbp.
            std::make_tuple(false, _0, _63, _0, _0, 0.0),
            // Unsorted, small numbers, many huge outliers -> good for k_wise_ns.
            std::make_tuple(false, _0, _63, largeVal, largeVal, 0.1),
            // Unsorted, small numbers, very rare outliers -> good for dynamic_vbp.
            std::make_tuple(false, _0, _63, largeVal, largeVal, 0.0001),
            // Unsorted, huge numbers in narrow range, no outliers -> good for for+dynamic_vbp.
            std::make_tuple(false, bitwidth_min<uint64_t>(63), bitwidth_min<uint64_t>(63) + 63, _0, _0, 0.0),
            // Sorted, large numbers -> good for delta+dynamic_vbp.
            std::make_tuple(true, _0, bitwidth_max<uint64_t>(24), _0, _0, 0.0),
            // Unsorted, random numbers -> good for nothing/uncompr.
            std::make_tuple(false, _0, bitwidth_min<uint64_t>(63), _0, _0, 0.0),
        }) {
            datasetIdx++;
            
            std::tie(isSorted, mainMin, mainMax, outlierMin, outlierMax, outlierShare) = params;
            const unsigned inMaxBw = effective_bitwidth((outlierShare > 0) ? outlierMax : mainMax);

#ifdef SELECT_BENCHMARK_2_TIME
            varex.print_datagen_started();
#else
            std::cerr << "generating input data column... ";
#endif
            auto inDataCol = generate_with_outliers_and_selectivity(
                    countValues,
                    mainMin, mainMax,
                    selectedShare,
                    outlierMin, outlierMax, outlierShare,
                    isSorted
            );
#ifdef SELECT_BENCHMARK_2_TIME
            varex.print_datagen_done();
            
            std::vector<varex_t::variant_t> variants;
            switch(inMaxBw) {
                // Generated with python:
                // for bw in range(1, 64+1):
                //   print("case {: >2}: variants = make_variants<varex_t, outMaxBw, {: >2}>(); break;".format(bw, bw))
                case  1: variants = make_variants<varex_t, outMaxBw,  1>(); break;
                case  2: variants = make_variants<varex_t, outMaxBw,  2>(); break;
                case  3: variants = make_variants<varex_t, outMaxBw,  3>(); break;
                case  4: variants = make_variants<varex_t, outMaxBw,  4>(); break;
                case  5: variants = make_variants<varex_t, outMaxBw,  5>(); break;
                case  6: variants = make_variants<varex_t, outMaxBw,  6>(); break;
                case  7: variants = make_variants<varex_t, outMaxBw,  7>(); break;
                case  8: variants = make_variants<varex_t, outMaxBw,  8>(); break;
                case  9: variants = make_variants<varex_t, outMaxBw,  9>(); break;
                case 10: variants = make_variants<varex_t, outMaxBw, 10>(); break;
                case 11: variants = make_variants<varex_t, outMaxBw, 11>(); break;
                case 12: variants = make_variants<varex_t, outMaxBw, 12>(); break;
                case 13: variants = make_variants<varex_t, outMaxBw, 13>(); break;
                case 14: variants = make_variants<varex_t, outMaxBw, 14>(); break;
                case 15: variants = make_variants<varex_t, outMaxBw, 15>(); break;
                case 16: variants = make_variants<varex_t, outMaxBw, 16>(); break;
                case 17: variants = make_variants<varex_t, outMaxBw, 17>(); break;
                case 18: variants = make_variants<varex_t, outMaxBw, 18>(); break;
                case 19: variants = make_variants<varex_t, outMaxBw, 19>(); break;
                case 20: variants = make_variants<varex_t, outMaxBw, 20>(); break;
                case 21: variants = make_variants<varex_t, outMaxBw, 21>(); break;
                case 22: variants = make_variants<varex_t, outMaxBw, 22>(); break;
                case 23: variants = make_variants<varex_t, outMaxBw, 23>(); break;
                case 24: variants = make_variants<varex_t, outMaxBw, 24>(); break;
                case 25: variants = make_variants<varex_t, outMaxBw, 25>(); break;
                case 26: variants = make_variants<varex_t, outMaxBw, 26>(); break;
                case 27: variants = make_variants<varex_t, outMaxBw, 27>(); break;
                case 28: variants = make_variants<varex_t, outMaxBw, 28>(); break;
                case 29: variants = make_variants<varex_t, outMaxBw, 29>(); break;
                case 30: variants = make_variants<varex_t, outMaxBw, 30>(); break;
                case 31: variants = make_variants<varex_t, outMaxBw, 31>(); break;
                case 32: variants = make_variants<varex_t, outMaxBw, 32>(); break;
                case 33: variants = make_variants<varex_t, outMaxBw, 33>(); break;
                case 34: variants = make_variants<varex_t, outMaxBw, 34>(); break;
                case 35: variants = make_variants<varex_t, outMaxBw, 35>(); break;
                case 36: variants = make_variants<varex_t, outMaxBw, 36>(); break;
                case 37: variants = make_variants<varex_t, outMaxBw, 37>(); break;
                case 38: variants = make_variants<varex_t, outMaxBw, 38>(); break;
                case 39: variants = make_variants<varex_t, outMaxBw, 39>(); break;
                case 40: variants = make_variants<varex_t, outMaxBw, 40>(); break;
                case 41: variants = make_variants<varex_t, outMaxBw, 41>(); break;
                case 42: variants = make_variants<varex_t, outMaxBw, 42>(); break;
                case 43: variants = make_variants<varex_t, outMaxBw, 43>(); break;
                case 44: variants = make_variants<varex_t, outMaxBw, 44>(); break;
                case 45: variants = make_variants<varex_t, outMaxBw, 45>(); break;
                case 46: variants = make_variants<varex_t, outMaxBw, 46>(); break;
                case 47: variants = make_variants<varex_t, outMaxBw, 47>(); break;
                case 48: variants = make_variants<varex_t, outMaxBw, 48>(); break;
                case 49: variants = make_variants<varex_t, outMaxBw, 49>(); break;
                case 50: variants = make_variants<varex_t, outMaxBw, 50>(); break;
                case 51: variants = make_variants<varex_t, outMaxBw, 51>(); break;
                case 52: variants = make_variants<varex_t, outMaxBw, 52>(); break;
                case 53: variants = make_variants<varex_t, outMaxBw, 53>(); break;
                case 54: variants = make_variants<varex_t, outMaxBw, 54>(); break;
                case 55: variants = make_variants<varex_t, outMaxBw, 55>(); break;
                case 56: variants = make_variants<varex_t, outMaxBw, 56>(); break;
                case 57: variants = make_variants<varex_t, outMaxBw, 57>(); break;
                case 58: variants = make_variants<varex_t, outMaxBw, 58>(); break;
                case 59: variants = make_variants<varex_t, outMaxBw, 59>(); break;
                case 60: variants = make_variants<varex_t, outMaxBw, 60>(); break;
                case 61: variants = make_variants<varex_t, outMaxBw, 61>(); break;
                case 62: variants = make_variants<varex_t, outMaxBw, 62>(); break;
                case 63: variants = make_variants<varex_t, outMaxBw, 63>(); break;
                case 64: variants = make_variants<varex_t, outMaxBw, 64>(); break;
            }
            
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
            MONITORING_ADD_INT_FOR("inMaxBw", inMaxBw, datasetIdx);
            MONITORING_ADD_INT_FOR("outMaxBw", outMaxBw, datasetIdx);
            
            // Data characteristics of the input data column.
            std::cerr << std::endl << "analyzing input data column... ";
            MONITORING_ADD_INT_FOR(
                    "inData_ValueCount",
                    inDataCol->get_count_values(),
                    datasetIdx
            );
            MONITORING_ADD_DATAPROPERTIES_FOR(
                    "inData", data_properties(inDataCol, false), datasetIdx
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
                    inDataCol->get_count_values(),
                    datasetIdx
            );
            MONITORING_ADD_DATAPROPERTIES_FOR(
                    "outPos", data_properties(inDataCol, false), datasetIdx
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
    
    return 0;
}