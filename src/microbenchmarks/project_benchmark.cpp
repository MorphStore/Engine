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

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/default_formats.h>
#include <core/morphing/delta.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/for.h>
#include <core/morphing/format.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/vbp.h>
#include <core/morphing/format_names.h> // Must be included after all formats.
#include <core/operators/general_vectorized/project_compr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/variant_executor.h>
#include <vector/vector_extension_names.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vectorlib;


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


// ****************************************************************************
// Main program.
// ****************************************************************************

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
    
    const size_t inDataCountA = 128 * 1024 * 1024;
    const size_t inDataCountB1 = 1024;
    const size_t inDataCountB2 = 2 * 1024 * 1024;
    
    const size_t inPosCountA1 = static_cast<size_t>(inDataCountA * 0.9);
    const size_t inPosCountA3 = static_cast<size_t>(inDataCountA * 0.01);
    const size_t inPosCountB = 128 * 1024 * 1024;
    
    // @todo It would be nice to use a 64-bit value, but then, some vector-lib
    // primitives would interpret it as a negative number. This would hurt,
    // e.g., FOR.
    const uint64_t largeVal = bitwidth_max<uint64_t>(63);
    // Looks strange, but saves us from casting in the initializer list.
    const uint64_t _0 = 0;
    const uint64_t _7 = 7;
    const uint64_t _63 = 63;
    const uint64_t min47bit = bitwidth_min<uint64_t>(47);
    const uint64_t max63bit = bitwidth_max<uint64_t>(63);
    const uint64_t range = 100000;
    
    std::vector<
            std::tuple<
                    size_t, bool, size_t,
                    bool, uint64_t, uint64_t, uint64_t, uint64_t, double
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
            std::make_tuple(false, _0, _7, _0, _0, 0.0), // C0
            std::make_tuple(false, _0, _63, _0, _0, 0.0), // C1
            std::make_tuple(false, _0, _63, largeVal, largeVal, 0.0001), // C2
            std::make_tuple(true, min47bit, min47bit + range, _0, _0, 0.0), // C5
            std::make_tuple(false, _0, max63bit, _0, _0, 0.0), // C6
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
            std::tie(
                    inDataSorted, inDataMainMin, inDataMainMax,
                    inDataOutlierMin, inDataOutlierMax, inDataOutlierShare
            ) = inDataCh;
            params.push_back(std::make_tuple(
                    inPosCount, inPosSorted,
                    inDataCount, inDataSorted, inDataMainMin, inDataMainMax,
                    inDataOutlierMin, inDataOutlierMax, inDataOutlierShare
            ));
        }
    
    // ========================================================================
    // Variant execution for each setting.
    // ========================================================================
    
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
    
    unsigned datasetIdx = 0;
    // @todo This results in a frequent regeneration of the same data (but it
    // does not seem to be a performance issue).
    for(auto param : params) {
        datasetIdx++;

        std::tie(
                inPosCount, inPosSorted, //inPosContiguous
                inDataCount, inDataSorted, inDataMainMin, inDataMainMax,
                inDataOutlierMin, inDataOutlierMax, inDataOutlierShare
        ) = param;
        const uint64_t inDataMaxVal = std::max(inDataMainMax, inDataOutlierMax);
        
        // --------------------------------------------------------------------
        // Data generation.
        // --------------------------------------------------------------------
        
        varex.print_datagen_started();
        auto inDataCol = generate_with_outliers_and_selectivity(
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
//            if(inPosContiguous) {
//                inPosCol = generate_sorted_unique(inPosCount);
//                inPosMaxVal = inPosCount - 1;
//            }
//            else {
                inPosCol = generate_sorted_unique_extraction(
                        inPosCount, inDataCount
                );
                inPosMaxVal = inDataCount - 1;
//            }
        }
        else {
            // Case B
            inPosCol = generate_with_distr(
                    inPosCount,
                    std::uniform_int_distribution<uint64_t>(0, inDataCount - 1),
                    false
            );
            inPosMaxVal = inDataCount - 1;
        }
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
        VG_CASE(largeVal        , inDataCountA - 1)
        VG_CASE(min47bit + range, inDataCountA - 1)
        VG_CASE(max63bit        , inDataCountA - 1)
        // Case B
        VG_CASE(_7              , inDataCountB1 - 1)
        VG_CASE(_63             , inDataCountB1 - 1)
        VG_CASE(largeVal        , inDataCountB1 - 1)
        VG_CASE(min47bit + range, inDataCountB1 - 1)
        VG_CASE(max63bit        , inDataCountB1 - 1)
        VG_CASE(_7              , inDataCountB2 - 1)
        VG_CASE(_63             , inDataCountB2 - 1)
        VG_CASE(largeVal        , inDataCountB2 - 1)
        VG_CASE(min47bit + range, inDataCountB2 - 1)
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