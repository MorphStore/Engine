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
 * @file select_sum_benchmark.cpp
 * @brief A micro benchmark of a simple query.
 * 
 * Assuming a relation `R` with two attributes `X` and `Y`, this micro
 * benchmark evaluates the behavior of the SQL query `SELECT SUM(Y) FROM R
 * WHERE X == c` for different characteristics of the base columns `X` and `Y`
 * and different formats for the base and intermediate columns.
 */

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
#include <core/morphing/default_formats.h>
#include <core/morphing/delta.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/for.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/vbp.h>
#include <core/morphing/format_names.h> // Must be included after all formats.
#include <core/operators/general_vectorized/agg_sum_compr.h>
#include <core/operators/general_vectorized/project_compr.h>
#include <core/operators/general_vectorized/select_compr.h>
#include <core/utils/variant_executor.h>

#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vectorlib;


// ****************************************************************************
// Query function.
// ****************************************************************************

#define KEYS_FOR_MONITORING \
    veName<t_ve>, \
    formatName<t_in_data_x_f>, formatName<t_in_data_y_f>, formatName<t_mid_pos_xc_f>, formatName<t_mid_data_yc_f>, \
    c, settingIdx

template<
        class t_vector_extension,
        class t_in_data_x_f,
        class t_in_data_y_f,
        class t_mid_pos_xc_f,
        class t_mid_data_yc_f
>
const column<uncompr_f> * select_sum_query(
        const column<t_in_data_x_f> * inDataXCol,
        const column<t_in_data_y_f> * inDataYCol,
        uint64_t c,
        MSV_CXX_ATTRIBUTE_PPUNUSED unsigned settingIdx
) {
    // SELECT SUM (Y) FROM R WHERE X == c
    
    using t_ve = t_vector_extension;
    
    inDataYCol->template prepare_for_random_access<t_ve>();
    
    MONITORING_START_INTERVAL_FOR("runtime select:µs", KEYS_FOR_MONITORING);
    auto midPosXCCol = my_select_wit_t<equal, t_ve, t_mid_pos_xc_f, t_in_data_x_f>::apply(inDataXCol, c);
    MONITORING_END_INTERVAL_FOR("runtime select:µs", KEYS_FOR_MONITORING);
    
    MONITORING_START_INTERVAL_FOR("runtime project:µs", KEYS_FOR_MONITORING);
    auto midDataYCCol = my_project_wit_t<t_ve, t_mid_data_yc_f, t_in_data_y_f, t_mid_pos_xc_f>::apply(inDataYCol, midPosXCCol);
    MONITORING_END_INTERVAL_FOR("runtime project:µs", KEYS_FOR_MONITORING);
    
    MONITORING_START_INTERVAL_FOR("runtime agg_sum:µs", KEYS_FOR_MONITORING);
    auto outDataCol = agg_sum<t_ve, t_mid_data_yc_f>(midDataYCCol);
    MONITORING_END_INTERVAL_FOR("runtime agg_sum:µs", KEYS_FOR_MONITORING);
    
    MONITORING_ADD_INT_FOR("inDataX_countValues", inDataXCol->get_count_values(), KEYS_FOR_MONITORING);
    MONITORING_ADD_INT_FOR("inDataX_sizeUsedByte", inDataXCol->get_size_used_byte(), KEYS_FOR_MONITORING);
    
    MONITORING_ADD_INT_FOR("inDataY_countValues", inDataYCol->get_count_values(), KEYS_FOR_MONITORING);
    MONITORING_ADD_INT_FOR("inDataY_sizeUsedByte", inDataYCol->get_size_used_byte(), KEYS_FOR_MONITORING);
    
    MONITORING_ADD_INT_FOR("midPosXC_countValues", midPosXCCol->get_count_values(), KEYS_FOR_MONITORING);
    MONITORING_ADD_INT_FOR("midPosXC_sizeUsedByte", midPosXCCol->get_size_used_byte(), KEYS_FOR_MONITORING);
    
    MONITORING_ADD_INT_FOR("midDataYC_countValues", midDataYCCol->get_count_values(), KEYS_FOR_MONITORING);
    MONITORING_ADD_INT_FOR("midDataYC_sizeUsedByte", midDataYCCol->get_size_used_byte(), KEYS_FOR_MONITORING);
    
    delete midPosXCCol;
    delete midDataYCCol;
    
    return outDataCol;
}


// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(ve, in_data_x_f, in_data_y_f, mid_pos_xc_f, mid_data_yc_f) { \
    new typename t_varex_t::operator_wrapper::template for_output_formats<uncompr_f>::template for_input_formats<in_data_x_f, in_data_y_f>( \
        &select_sum_query<ve, in_data_x_f, in_data_y_f, mid_pos_xc_f, mid_data_yc_f> \
    ), \
    veName<ve>, \
    formatName<in_data_x_f>, \
    formatName<in_data_y_f>, \
    formatName<mid_pos_xc_f>, \
    formatName<mid_data_yc_f> \
}

#if 0 // all variants (compressed and uncompressed)
#define MAKE_VARIANTS_VARY_MIDDATAYCF(ve, in_data_x_f, in_data_y_f, mid_pos_xc_f, midDataYCBw) \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(in_data_x_f), \
            SINGLE_ARG(in_data_y_f), \
            SINGLE_ARG(mid_pos_xc_f), \
            uncompr_f \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(in_data_x_f), \
            SINGLE_ARG(in_data_y_f), \
            SINGLE_ARG(mid_pos_xc_f), \
            DEFAULT_STATIC_VBP_F(ve, midDataYCBw) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(in_data_x_f), \
            SINGLE_ARG(in_data_y_f), \
            SINGLE_ARG(mid_pos_xc_f), \
            DEFAULT_DYNAMIC_VBP_F(ve) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(in_data_x_f), \
            SINGLE_ARG(in_data_y_f), \
            SINGLE_ARG(mid_pos_xc_f), \
            DEFAULT_DELTA_DYNAMIC_VBP_F(ve) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(in_data_x_f), \
            SINGLE_ARG(in_data_y_f), \
            SINGLE_ARG(mid_pos_xc_f), \
            DEFAULT_FOR_DYNAMIC_VBP_F(ve) \
    )
#define MAKE_VARIANTS_VARY_MIDPOSXCF(ve, in_data_x_f, in_data_y_f, midPosXCBw, midDataYCBw) \
    MAKE_VARIANTS_VARY_MIDDATAYCF( \
            ve, \
            SINGLE_ARG(in_data_x_f), \
            SINGLE_ARG(in_data_y_f), \
            uncompr_f, \
            midDataYCBw \
    ), \
    MAKE_VARIANTS_VARY_MIDDATAYCF( \
            ve, \
            SINGLE_ARG(in_data_x_f), \
            SINGLE_ARG(in_data_y_f), \
            DEFAULT_STATIC_VBP_F(ve, midPosXCBw), \
            midDataYCBw \
    ), \
    MAKE_VARIANTS_VARY_MIDDATAYCF( \
            ve, \
            SINGLE_ARG(in_data_x_f), \
            SINGLE_ARG(in_data_y_f), \
            DEFAULT_DYNAMIC_VBP_F(ve), \
            midDataYCBw \
    ), \
    MAKE_VARIANTS_VARY_MIDDATAYCF( \
            ve, \
            SINGLE_ARG(in_data_x_f), \
            SINGLE_ARG(in_data_y_f), \
            DEFAULT_DELTA_DYNAMIC_VBP_F(ve), \
            midDataYCBw \
    ), \
    MAKE_VARIANTS_VARY_MIDDATAYCF( \
            ve, \
            SINGLE_ARG(in_data_x_f), \
            SINGLE_ARG(in_data_y_f), \
            DEFAULT_FOR_DYNAMIC_VBP_F(ve), \
            midDataYCBw \
    )
// This column needs random access, therefore only the two supported formats.
#define MAKE_VARIANTS_VARY_INDATAYF(ve, in_data_x_f, inDataYBw, midPosXCBw, midDataYCBw) \
    MAKE_VARIANTS_VARY_MIDPOSXCF( \
            ve, \
            SINGLE_ARG(in_data_x_f), \
            uncompr_f, \
            midPosXCBw, \
            midDataYCBw \
    ), \
    MAKE_VARIANTS_VARY_MIDPOSXCF( \
            ve, \
            SINGLE_ARG(in_data_x_f), \
            DEFAULT_STATIC_VBP_F(ve, inDataYBw), \
            midPosXCBw, \
            midDataYCBw \
    )
#define MAKE_VARIANTS_VARY_INDATAXF(ve, inDataXBw, inDataYBw, midPosXCBw, midDataYCBw) \
    MAKE_VARIANTS_VARY_INDATAYF( \
            ve, \
            uncompr_f, \
            inDataYBw, \
            midPosXCBw, \
            midDataYCBw \
    ), \
    MAKE_VARIANTS_VARY_INDATAYF( \
            ve, \
            DEFAULT_STATIC_VBP_F(ve, inDataXBw), \
            inDataYBw, \
            midPosXCBw, \
            midDataYCBw \
    ), \
    MAKE_VARIANTS_VARY_INDATAYF( \
            ve, \
            DEFAULT_DYNAMIC_VBP_F(ve), \
            inDataYBw, \
            midPosXCBw, \
            midDataYCBw \
    ), \
    MAKE_VARIANTS_VARY_INDATAYF( \
            ve, \
            DEFAULT_DELTA_DYNAMIC_VBP_F(ve), \
            inDataYBw, \
            midPosXCBw, \
            midDataYCBw \
    ), \
    MAKE_VARIANTS_VARY_INDATAYF( \
            ve, \
            DEFAULT_FOR_DYNAMIC_VBP_F(ve), \
            inDataYBw, \
            midPosXCBw, \
            midDataYCBw \
    )
#elif 1 // only some selected variants
#define MAKE_VARIANTS_VARY_INDATAXF(ve, inDataXBw, inDataYBw, midPosXCBw, midDataYCBw) \
    MAKE_VARIANT( \
            ve, \
            uncompr_f, \
            uncompr_f, \
            uncompr_f, \
            uncompr_f \
    ), \
    MAKE_VARIANT( \
            ve, \
            DEFAULT_STATIC_VBP_F(ve, inDataXBw), \
            DEFAULT_STATIC_VBP_F(ve, inDataYBw), \
            uncompr_f, \
            uncompr_f \
    ), \
    MAKE_VARIANT( \
            ve, \
            DEFAULT_STATIC_VBP_F(ve, inDataXBw), \
            DEFAULT_STATIC_VBP_F(ve, inDataYBw), \
            DEFAULT_STATIC_VBP_F(ve, midPosXCBw), \
            DEFAULT_STATIC_VBP_F(ve, midDataYCBw) \
    ), \
    MAKE_VARIANT( \
            ve, \
            DEFAULT_STATIC_VBP_F(ve, inDataXBw), \
            DEFAULT_STATIC_VBP_F(ve, inDataYBw), \
            DEFAULT_DELTA_DYNAMIC_VBP_F(ve), \
            DEFAULT_DELTA_DYNAMIC_VBP_F(ve) \
    ), \
    MAKE_VARIANT( \
            ve, \
            DEFAULT_STATIC_VBP_F(ve, inDataXBw), \
            DEFAULT_STATIC_VBP_F(ve, inDataYBw), \
            DEFAULT_FOR_DYNAMIC_VBP_F(ve), \
            DEFAULT_FOR_DYNAMIC_VBP_F(ve) \
    )
#else // uncompressed only
#define MAKE_VARIANTS_VARY_INDATAXF(ve, inDataXBw, inDataYBw, midPosXCBw, midDataYCBw) \
    MAKE_VARIANT( \
            ve, \
            uncompr_f, \
            uncompr_f, \
            uncompr_f, \
            uncompr_f \
    )
#endif

template<
        class t_varex_t,
        unsigned t_InDataXBw,
        unsigned t_InDataYBw,
        unsigned t_MidPosXCBw,
        unsigned t_MidDataYCBw
>
std::vector<typename t_varex_t::variant_t> make_variants() {
    return {
#ifdef AVX512
        MAKE_VARIANTS_VARY_INDATAXF(
                avx512<v512<uint64_t>>,
                t_InDataXBw, t_InDataYBw, t_MidPosXCBw, t_MidDataYCBw
        ),
#elif defined(AVXTWO)
        MAKE_VARIANTS_VARY_INDATAXF(
                avx2<v256<uint64_t>>,
                t_InDataXBw, t_InDataYBw, t_MidPosXCBw, t_MidDataYCBw
        ),
#elif defined(SSE)
        MAKE_VARIANTS_VARY_INDATAXF(
                sse<v128<uint64_t>>,
                t_InDataXBw, t_InDataYBw, t_MidPosXCBw, t_MidDataYCBw
        ),
#else
        MAKE_VARIANTS_VARY_INDATAXF(
                scalar<v64<uint64_t>>,
                t_InDataXBw, t_InDataYBw, t_MidPosXCBw, t_MidDataYCBw
        ),
#endif
    };
}

#define VG_BEGIN \
    if(false) {/*dummy*/}
#define VG_CASE(_inDataXMaxVal, _inDataYMaxVal) \
    else if(inDataXMaxVal == _inDataXMaxVal && inDataYMaxVal == _inDataYMaxVal) \
        variants = make_variants< \
                varex_t, \
                effective_bitwidth(_inDataXMaxVal), \
                effective_bitwidth(_inDataYMaxVal), \
                effective_bitwidth(inDataCount - 1), \
                effective_bitwidth(_inDataYMaxVal) \
        >();
#define VG_END \
    else throw std::runtime_error( \
            "unexpected combination: " \
            "inDataXMaxVal=" + std::to_string(inDataXMaxVal) + ", " \
            "inDataYMaxVal=" + std::to_string(inDataYMaxVal) \
    );


// ****************************************************************************
// Main program.
// ****************************************************************************

const size_t inDataCount = 128 * 1024 * 1024;

// @todo It would be nice to use a 64-bit value, but then, some TVL primitives
//       would interpret it as a negative number. This would hurt, e.g., FOR.
// Looks strange, but saves us from casting below.
const uint64_t _0 = 0;
const uint64_t _7 = 7;
const uint64_t _63 = 63;
const uint64_t _100k = 100000;
const uint64_t min48bit = bitwidth_min<uint64_t>(48);
const uint64_t min63bit = bitwidth_min<uint64_t>(63);
const uint64_t max63bit = bitwidth_max<uint64_t>(63);

using col_param_t = std::tuple<
        bool, uint64_t, uint64_t, uint64_t, uint64_t, double
>;
using param_t = std::tuple<col_param_t, col_param_t>;

std::vector<param_t> get_params() {
    std::vector<param_t> params;
    
    // Cx (Cy): number in PD's PhD thesis (number in our PVLDB-paper).
    auto c1 = std::make_tuple(false, _0, _63, _0, _0, 0.0); // C2 (C1)
    auto c2 = std::make_tuple(false, _0, _63, max63bit, max63bit, 0.0001); // C3 (C2)
    auto c3 = std::make_tuple(false, min63bit, min63bit + 63, _0, _0, 0.0); // C4 (C3)
    auto c4 = std::make_tuple(true, min48bit, min48bit + _100k, _0, _0, 0.0); // C6 (C4)
    
    // Dummy to ensure warm-start.
    params.push_back(std::make_tuple(c1, c1));
    
    params.push_back(std::make_tuple(c1, c1)); // case 1
    params.push_back(std::make_tuple(c1, c4)); // case 2 
    params.push_back(std::make_tuple(c2, c3)); // case 3
    
    return params;
}

std::tuple<const column<uncompr_f> *, uint64_t, uint64_t> generate_col(
        col_param_t param, double selectivity
) {
    bool isSorted;
    uint64_t mainMin;
    uint64_t mainMax;
    uint64_t outlierMin;
    uint64_t outlierMax;
    double outlierShare;
    std::tie(
            isSorted, mainMin, mainMax, outlierMin, outlierMax, outlierShare
    ) = param;
    
    auto col= ColumnGenerator::generate_with_outliers_and_selectivity(
            inDataCount,
            mainMin, mainMax, selectivity,
            outlierMin, outlierMax, outlierShare,
            isSorted
    );

    return std::make_tuple(col, mainMin, std::max(mainMax, outlierMax));
}

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    // ========================================================================
    // Creation of the variant executor.
    // ========================================================================
    
    using varex_t = variant_executor_helper<1, 2, uint64_t, unsigned>::type
        ::for_variant_params<
                std::string, std::string, std::string, std::string, std::string
        >
        ::for_setting_params<>;
    varex_t varex(
            {"predicate", "settingIdx"},
            {
                "vector_extension",
                "in_data_x_f", "in_data_y_f", "mid_pos_xc_f", "mid_data_yc_f"
            },
            {}
    );
    
    // ========================================================================
    // Specification of the settings.
    // ========================================================================
    
    auto params = get_params();
    
    // ========================================================================
    // Variant execution for each setting.
    // ========================================================================
    
    unsigned settingIdx = 0;
    for(auto param : params) {
        settingIdx++;
        
        // --------------------------------------------------------------------
        // Data generation.
        // --------------------------------------------------------------------
        
        varex.print_datagen_started();
        const column<uncompr_f> * inDataXCol;
        const column<uncompr_f> * inDataYCol;
        uint64_t inDataXMainMin;
        uint64_t inDataXMaxVal;
        uint64_t inDataYMaxVal;
        std::tie(inDataXCol, inDataXMainMin, inDataXMaxVal) =
                generate_col(std::get<0>(param), 0.9);
        std::tie(inDataYCol, std::ignore, inDataYMaxVal) =
                generate_col(std::get<1>(param), 0);
        varex.print_datagen_done();
        
        // --------------------------------------------------------------------
        // Variant generation.
        // --------------------------------------------------------------------
        
        std::vector<varex_t::variant_t> variants;
        
        VG_BEGIN
        VG_CASE(_63, _63)
        VG_CASE(_63, min48bit + _100k)
        VG_CASE(max63bit, min63bit + 63)
        VG_END
        
        // --------------------------------------------------------------------
        // Variant execution.
        // --------------------------------------------------------------------
        
        varex.execute_variants(
                variants, inDataXCol, inDataYCol, inDataXMainMin, settingIdx
        );

        delete inDataXCol;
        delete inDataYCol;
    }
    
    varex.done();
    
    return !varex.good();
}
