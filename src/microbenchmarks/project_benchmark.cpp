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
 * @file project_benchmark.cpp
 * @brief A micro benchmark of the project-operator.
 */

//#define PROJECT_BENCHMARK_TIME

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
#ifdef PROJECT_BENCHMARK_TIME
#include <core/morphing/default_formats.h>
#include <core/morphing/delta.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/for.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/vbp.h>
#include <core/morphing/format_names.h> // Must be included after all formats.
#include <core/operators/general_vectorized/project_compr.h>
#include <core/utils/variant_executor.h>
#else
#include <core/operators/scalar/project_uncompr.h>
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


#ifdef PROJECT_BENCHMARK_TIME

// ****************************************************************************
// Amendment for the format names (just for this benchmark).
// ****************************************************************************
// A special format name for static_vbp_f with a bit width divisible by eight
// (byte-packing).

#define SPECIALIZE_FORMATNAME_BYTEPACKING(bw) \
    template<unsigned t_Step> \
    std::string formatName< \
            static_vbp_f<vbp_l<bw, t_Step> > \
    > = "static_vbp_byte_f<vbp_l<bw, " + std::to_string(t_Step) + "> >";

SPECIALIZE_FORMATNAME_BYTEPACKING( 8)
SPECIALIZE_FORMATNAME_BYTEPACKING(16)
SPECIALIZE_FORMATNAME_BYTEPACKING(24)
SPECIALIZE_FORMATNAME_BYTEPACKING(32)
SPECIALIZE_FORMATNAME_BYTEPACKING(40)
SPECIALIZE_FORMATNAME_BYTEPACKING(48)
SPECIALIZE_FORMATNAME_BYTEPACKING(56)
SPECIALIZE_FORMATNAME_BYTEPACKING(64)

#undef SPECIALIZE_FORMATNAME_BYTEPACKING

// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(ve, out_data_f, in_data_f, in_pos_f) { \
    new typename t_varex_t::operator_wrapper::template for_output_formats<out_data_f>::template for_input_formats<in_data_f, in_pos_f>( \
        &my_project_wit_t<ve, out_data_f, in_data_f, in_pos_f>::apply \
    ), \
    veName<ve>, \
    formatName<out_data_f>, \
    formatName<in_data_f>, \
    formatName<in_pos_f> \
}

#if 1 // all variants (compressed and uncompressed)
#define MAKE_VARIANTS_VE_OUTDATA_INDATA(ve, out_data_f, in_data_f, inPosBw) \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_data_f), \
            SINGLE_ARG(in_data_f), \
            uncompr_f \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_data_f), \
            SINGLE_ARG(in_data_f), \
            DEFAULT_STATIC_VBP_F(ve, inPosBw) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_data_f), \
            SINGLE_ARG(in_data_f), \
            DEFAULT_DYNAMIC_VBP_F(ve) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_data_f), \
            SINGLE_ARG(in_data_f), \
            DEFAULT_DELTA_DYNAMIC_VBP_F(ve) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_data_f), \
            SINGLE_ARG(in_data_f), \
            DEFAULT_FOR_DYNAMIC_VBP_F(ve) \
    )

// Note that, up to now, random access is only supported to uncompr_f and
// static_vbp_f. Therefore, we have only these options here.
#define MAKE_VARIANTS_VE_OUTDATA(ve, out_data_f, inDataBw, inPosBw) \
    MAKE_VARIANTS_VE_OUTDATA_INDATA( \
            ve, \
            SINGLE_ARG(out_data_f), \
            uncompr_f, \
            inPosBw \
    ), \
    MAKE_VARIANTS_VE_OUTDATA_INDATA( \
            ve, \
            SINGLE_ARG(out_data_f), \
            DEFAULT_STATIC_VBP_F(ve, inDataBw), \
            inPosBw \
    ), \
    MAKE_VARIANTS_VE_OUTDATA_INDATA( \
            ve, \
            SINGLE_ARG(out_data_f), \
            DEFAULT_STATIC_VBP_BYTE_F(ve, inDataBw), \
            inPosBw \
    )

#define MAKE_VARIANTS_VE(ve, outDataBw, inDataBw, inPosBw) \
    MAKE_VARIANTS_VE_OUTDATA( \
            ve, \
            uncompr_f, \
            inDataBw, \
            inPosBw \
    ), \
    MAKE_VARIANTS_VE_OUTDATA( \
            ve, \
            DEFAULT_STATIC_VBP_F(ve, outDataBw), \
            inDataBw, \
            inPosBw \
    ), \
    MAKE_VARIANTS_VE_OUTDATA( \
            ve, \
            DEFAULT_DYNAMIC_VBP_F(ve), \
            inDataBw, \
            inPosBw \
    ), \
    MAKE_VARIANTS_VE_OUTDATA( \
            ve, \
            DEFAULT_DELTA_DYNAMIC_VBP_F(ve), \
            inDataBw, \
            inPosBw \
    ), \
    MAKE_VARIANTS_VE_OUTDATA( \
            ve, \
            DEFAULT_FOR_DYNAMIC_VBP_F(ve), \
            inDataBw, \
            inPosBw \
    )
#else // uncompressed only
#define MAKE_VARIANTS_VE(ve, outDataBw, inDataBw, inPosBw) \
    MAKE_VARIANT( \
            ve, \
            uncompr_f, \
            uncompr_f, \
            uncompr_f \
    )
#endif

template<class t_varex_t, unsigned t_OutDataBw, unsigned t_InDataBw, unsigned t_InPosBw>
std::vector<typename t_varex_t::variant_t> make_variants() {
    return {
        // Compressed variants.
        MAKE_VARIANTS_VE(scalar<v64<uint64_t>>, t_OutDataBw, t_InDataBw, t_InPosBw),
#ifdef SSE
        MAKE_VARIANTS_VE(sse<v128<uint64_t>>, t_OutDataBw, t_InDataBw, t_InPosBw),
#endif
#ifdef AVXTWO
        MAKE_VARIANTS_VE(avx2<v256<uint64_t>>, t_OutDataBw, t_InDataBw, t_InPosBw),
#endif
#ifdef AVX512
        MAKE_VARIANTS_VE(avx512<v512<uint64_t>>, t_OutDataBw, t_InDataBw, t_InPosBw),
#endif
    };
}

#define VG_BEGIN \
    if(false) {/*dummy*/}
#define VG_CASE(_inDataMaxVal, _inPosMaxVal) \
    else if(inDataMaxVal == _inDataMaxVal && inPosMaxVal == _inPosMaxVal) \
        variants = make_variants< \
                varex_t, \
                effective_bitwidth(_inDataMaxVal), \
                effective_bitwidth(_inDataMaxVal), \
                effective_bitwidth(_inPosMaxVal) \
        >();
#define VG_END \
    else throw std::runtime_error( \
            "unexpected combination: inDataMaxVal=" + \
            std::to_string(inDataMaxVal) + ", inPosMaxVal=" + \
            std::to_string(inPosMaxVal) \
    );

#endif


// ****************************************************************************
// Main program.
// ****************************************************************************

const size_t inDataCountA = 128 * 1024 * 1024;
const size_t inDataCountB1 = 1024;
const size_t inDataCountB2 = 2 * 1024 * 1024;

const size_t inPosCountA1 = static_cast<size_t>(inDataCountA * 0.9);
const size_t inPosCountA3 = static_cast<size_t>(inDataCountA * 0.01);
const size_t inPosCountB = 128 * 1024 * 1024;

// @todo It would be nice to use a 64-bit value, but then, some vector-lib
// primitives would interpret it as a negative number. This would hurt,
// e.g., FOR.
// Looks strange, but saves us from casting in the initializer list.
const uint64_t _0 = 0;
const uint64_t _7 = 7;
const uint64_t _63 = 63;
const uint64_t _100k = 100000;
const uint64_t min47bit = bitwidth_min<uint64_t>(47);
const uint64_t min48bit = bitwidth_min<uint64_t>(48);
const uint64_t max63bit = bitwidth_max<uint64_t>(63);

std::vector<std::tuple<
        size_t, bool, size_t,
        bool, uint64_t, uint64_t, uint64_t, uint64_t, double, bool
>> get_params() {
    std::vector<
            std::tuple<
                    size_t, bool, size_t,
                    bool, uint64_t, uint64_t, uint64_t, uint64_t, double, bool
            >
    > params;
    
    for(auto inCase : {
        // Case A
        std::make_tuple(inPosCountA1, true, inDataCountA),
        std::make_tuple(inPosCountA3, true, inDataCountA),
        // Case B
        std::make_tuple(inPosCountB, false, inDataCountB1),
        std::make_tuple(inPosCountB, false, inDataCountB2),
    })
        for(auto inDataCh : {
            std::make_tuple(false, _0, _7, _0, _0, 0.0, false), // C1
            std::make_tuple(false, _0, _63, _0, _0, 0.0, false), // C2
            std::make_tuple(false, _0, _63, max63bit, max63bit, 0.0001, false), // C3
            std::make_tuple(true, min48bit, min48bit + _100k, _0, _0, 0.0, false), // C6
            std::make_tuple(true, min47bit, min47bit + _100k, _0, _0, 0.0, false), // C6b
            std::make_tuple(false, _0, max63bit, _0, _0, 0.0, true), // C7
        }) {
            size_t inPosCount;
            bool inPosSorted;
            size_t inDataCount;
            std::tie(inPosCount, inPosSorted, inDataCount) = inCase;
            bool inDataSorted;
            uint64_t inDataMainMin;
            uint64_t inDataMainMax;
            uint64_t inDataOutlierMin;
            uint64_t inDataOutlierMax;
            double inDataOutlierShare;
            bool inDataAssumedUnique;
            std::tie(
                    inDataSorted, inDataMainMin, inDataMainMax,
                    inDataOutlierMin, inDataOutlierMax, inDataOutlierShare, inDataAssumedUnique
            ) = inDataCh;
            params.push_back(std::make_tuple(
                    inPosCount, inPosSorted,
                    inDataCount, inDataSorted, inDataMainMin, inDataMainMax,
                    inDataOutlierMin, inDataOutlierMax, inDataOutlierShare, inDataAssumedUnique
            ));
        }
        
    return params;
}

std::tuple<
        const column<uncompr_f> *,
        const column<uncompr_f> *,
        uint64_t,
        unsigned
> generate_data(std::tuple<
        size_t, bool, size_t,
        bool, uint64_t, uint64_t, uint64_t, uint64_t, double, bool
> param) {
    size_t inPosCount;
    size_t inDataCount;
    bool inDataSorted;
    uint64_t inDataMainMin;
    uint64_t inDataMainMax;
    uint64_t inDataOutlierMin;
    uint64_t inDataOutlierMax;
    double inDataOutlierShare;
    bool inPosSorted;
//    bool inPosContiguous;
    
    std::tie(
            inPosCount, inPosSorted, //inPosContiguous
            inDataCount, inDataSorted, inDataMainMin, inDataMainMax,
            inDataOutlierMin, inDataOutlierMax, inDataOutlierShare, std::ignore
    ) = param;

    const uint64_t inDataMaxVal = std::max(inDataMainMax, inDataOutlierMax);

    auto inDataCol = ColumnGenerator::generate_with_outliers_and_selectivity(
            inDataCount,
            inDataMainMin, inDataMainMax,
            0,
            inDataOutlierMin, inDataOutlierMax,
            inDataOutlierShare,
            inDataSorted
    );
    const column<uncompr_f> * inPosCol;
    unsigned inPosMaxVal;
    if(inPosSorted) {
        // Case A
//        if(inPosContiguous) {
//            inPosCol = generate_sorted_unique(inPosCount);
//            inPosMaxVal = inPosCount - 1;
//        }
//        else {
            inPosCol = ColumnGenerator::generate_sorted_unique_extraction(
                    inPosCount, inDataCount
            );
            inPosMaxVal = inDataCount - 1;
//        }
    }
    else {
        // Case B
        inPosCol = ColumnGenerator::generate_with_distr(
                inPosCount,
                std::uniform_int_distribution<uint64_t>(0, inDataCount - 1),
                false
        );
        inPosMaxVal = inDataCount - 1;
    }

    return std::make_tuple(inDataCol, inPosCol, inDataMaxVal, inPosMaxVal);
}

#ifdef PROJECT_BENCHMARK_TIME
int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    // ========================================================================
    // Creation of the variant executor.
    // ========================================================================
    
    using varex_t = variant_executor_helper<1, 2>::type
        ::for_variant_params<std::string, std::string, std::string, std::string>
        ::for_setting_params<unsigned>;
    varex_t varex(
            {},
            {"vector_extension", "out_data_f", "in_data_f", "in_pos_f"},
            {"datasetIdx"}
    );
    
    // ========================================================================
    // Specification of the settings.
    // ========================================================================
    
    auto params = get_params();
    
    // ========================================================================
    // Variant execution for each setting.
    // ========================================================================
    
    unsigned datasetIdx = 0;
    // @todo This results in a frequent regeneration of the same data (but it
    // does not seem to be a performance issue).
    for(auto param : params) {
        datasetIdx++;
        
        // --------------------------------------------------------------------
        // Data generation.
        // --------------------------------------------------------------------
        
        varex.print_datagen_started();
        const column<uncompr_f> * inDataCol;
        const column<uncompr_f> * inPosCol;
        uint64_t inDataMaxVal;
        unsigned inPosMaxVal; // @todo Why unsigned? Why not uint64_t?
        std::tie(inDataCol, inPosCol, inDataMaxVal, inPosMaxVal) = 
                generate_data(param);
        varex.print_datagen_done();
        
        // --------------------------------------------------------------------
        // Variant generation.
        // --------------------------------------------------------------------
        
        std::vector<varex_t::variant_t> variants;
        
        // Only enumerate the maximum bit widths that might actually be
        // encountered depending on the parameters of the data generation.
        // We do not need all 64 bit widths. This greatly reduces the
        // compilation time.
        VG_BEGIN
        // Case A
        VG_CASE(_7              , inDataCountA - 1)
        VG_CASE(_63             , inDataCountA - 1)
        VG_CASE(max63bit        , inDataCountA - 1)
        VG_CASE(min48bit + _100k, inDataCountA - 1)
        VG_CASE(min47bit + _100k, inDataCountA - 1)
        VG_CASE(max63bit        , inDataCountA - 1)
        // Case B
        VG_CASE(_7              , inDataCountB1 - 1)
        VG_CASE(_63             , inDataCountB1 - 1)
        VG_CASE(max63bit        , inDataCountB1 - 1)
        VG_CASE(min48bit + _100k, inDataCountB1 - 1)
        VG_CASE(min47bit + _100k, inDataCountB1 - 1)
        VG_CASE(max63bit        , inDataCountB1 - 1)
        VG_CASE(_7              , inDataCountB2 - 1)
        VG_CASE(_63             , inDataCountB2 - 1)
        VG_CASE(max63bit        , inDataCountB2 - 1)
        VG_CASE(min48bit + _100k, inDataCountB2 - 1)
        VG_CASE(min47bit + _100k, inDataCountB2 - 1)
        VG_CASE(max63bit        , inDataCountB2 - 1)
        VG_END
        
        // --------------------------------------------------------------------
        // Variant execution.
        // --------------------------------------------------------------------
        
        varex.execute_variants(variants, datasetIdx, inDataCol, inPosCol);

        delete inDataCol;
        delete inPosCol;
    }
    
    varex.done();
    
    return 0;
}
#else
int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    // ========================================================================
    // Specification of the settings.
    // ========================================================================
    
    auto params = get_params();
    
    // ========================================================================
    // Analysis of the input and output data.
    // ========================================================================
    
    unsigned datasetIdx = 0;
    // @todo This results in a frequent regeneration of the same data (but it
    // does not seem to be a performance issue).
    for(auto param : params) {
        datasetIdx++;
        std::cerr << "dataset " << datasetIdx << '/' << params.size() << std::endl;
        
        // The way we generate the data, sorted positions columns are always
        // unique.
        MSV_CXX_ATTRIBUTE_PPUNUSED const bool inPosAssumedUnique = std::get<1>(param);
        MSV_CXX_ATTRIBUTE_PPUNUSED const bool inDataAssumedUnique = std::get<9>(param);
        
        // --------------------------------------------------------------------
        // Data generation.
        // --------------------------------------------------------------------
        
        std::cerr << "\tgenerating input data... ";
        const column<uncompr_f> * inDataCol;
        const column<uncompr_f> * inPosCol;
        std::tie(inDataCol, inPosCol, std::ignore, std::ignore) = 
                generate_data(param);
        std::cerr << "done." << std::endl;
        
        // --------------------------------------------------------------------
        // Analysis of the input and output data.
        // --------------------------------------------------------------------
        
        MONITORING_CREATE_MONITOR(
                MONITORING_MAKE_MONITOR(datasetIdx),
                MONITORING_KEY_IDENTS("datasetIdx")
        );

        // Execution of the project-operator.
        std::cerr << "\texecuting project-operator... ";
        auto outDataCol = project<scalar<v64<uint64_t>>, uncompr_f>(
                inDataCol, inPosCol
        );
        std::cerr << "done." << std::endl;
        
        // Data characteristics of the input and output columns.
        for(auto colInfo : {
            std::make_tuple("inData" , inDataCol , inDataAssumedUnique),
            std::make_tuple("inPos"  , inPosCol  , inPosAssumedUnique),
            std::make_tuple("outData", outDataCol, inPosAssumedUnique ? inDataAssumedUnique : false),
        }) {
            std::string colName;
            const column<uncompr_f> * col;
            bool assumedUnique;
            std::tie(colName, col, assumedUnique) = colInfo;
            
            std::cerr << "\tanalyzing " << colName << " column... ";
            // @todo The data analysis in data_properties should also cover the
            // number of data elements, just for ease of use.
            MONITORING_ADD_INT_FOR(
                    colName + "_ValueCount",
                    col->get_count_values(),
                    datasetIdx
            );
            MONITORING_ADD_DATAPROPERTIES_FOR(
                    colName + "_", data_properties(col, assumedUnique), datasetIdx
            );
            std::cerr << "done." << std::endl;
            
            delete col;
        }
    }
    
    MONITORING_PRINT_MONITORS(monitorCsvLog);
    
    return 0;
}
#endif
