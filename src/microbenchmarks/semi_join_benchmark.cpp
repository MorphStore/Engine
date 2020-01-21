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
 * @file semi_join_benchmark.cpp
 * @brief A micro benchmark of the semi_join-operator.
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
#include <core/operators/general_vectorized/join_semi_equi_compr.h>
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
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(ve, out_pos_r_f, in_data_l_f, in_data_r_f) { \
    new typename t_varex_t::operator_wrapper::template for_output_formats<out_pos_r_f>::template for_input_formats<in_data_l_f, in_data_r_f>( \
        &semi_join<ve, out_pos_r_f, in_data_l_f, in_data_r_f> \
    ), \
    veName<ve>, \
    formatName<out_pos_r_f>, \
    formatName<in_data_l_f>, \
    formatName<in_data_r_f> \
}

#if 1 // all variants (compressed and uncompressed)
#define MAKE_VARIANTS_VE_OUTPOSR_INDATAL(ve, out_pos_r_f, in_data_l_f, inDataRBw) \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_r_f), \
            SINGLE_ARG(in_data_l_f), \
            uncompr_f \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_r_f), \
            SINGLE_ARG(in_data_l_f), \
            DEFAULT_STATIC_VBP_F(ve, inDataRBw) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_r_f), \
            SINGLE_ARG(in_data_l_f), \
            DEFAULT_DYNAMIC_VBP_F(ve) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_r_f), \
            SINGLE_ARG(in_data_l_f), \
            DEFAULT_DELTA_DYNAMIC_VBP_F(ve) \
    ), \
    MAKE_VARIANT( \
            ve, \
            SINGLE_ARG(out_pos_r_f), \
            SINGLE_ARG(in_data_l_f), \
            DEFAULT_FOR_DYNAMIC_VBP_F(ve) \
    )

#define MAKE_VARIANTS_VE_OUTPOSR(ve, out_pos_r_f, inDataLBw, inDataRBw) \
    MAKE_VARIANTS_VE_OUTPOSR_INDATAL( \
            ve, \
            SINGLE_ARG(out_pos_r_f), \
            uncompr_f, \
            inDataRBw \
    ), \
    MAKE_VARIANTS_VE_OUTPOSR_INDATAL( \
            ve, \
            SINGLE_ARG(out_pos_r_f), \
            DEFAULT_STATIC_VBP_F(ve, inDataLBw), \
            inDataRBw \
    ), \
    MAKE_VARIANTS_VE_OUTPOSR_INDATAL( \
            ve, \
            SINGLE_ARG(out_pos_r_f), \
            DEFAULT_DYNAMIC_VBP_F(ve), \
            inDataRBw \
    ), \
    MAKE_VARIANTS_VE_OUTPOSR_INDATAL( \
            ve, \
            SINGLE_ARG(out_pos_r_f), \
            DEFAULT_DELTA_DYNAMIC_VBP_F(ve), \
            inDataRBw \
    ), \
    MAKE_VARIANTS_VE_OUTPOSR_INDATAL( \
            ve, \
            SINGLE_ARG(out_pos_r_f), \
            DEFAULT_FOR_DYNAMIC_VBP_F(ve), \
            inDataRBw \
    )

#define MAKE_VARIANTS_VE(ve, outPosRBw, inDataLBw, inDataRBw) \
    MAKE_VARIANTS_VE_OUTPOSR( \
            ve, \
            uncompr_f, \
            inDataLBw, \
            inDataRBw \
    ), \
    MAKE_VARIANTS_VE_OUTPOSR( \
            ve, \
            DEFAULT_STATIC_VBP_F(ve, outPosRBw), \
            inDataLBw, \
            inDataRBw \
    ), \
    MAKE_VARIANTS_VE_OUTPOSR( \
            ve, \
            DEFAULT_DYNAMIC_VBP_F(ve), \
            inDataLBw, \
            inDataRBw \
    ), \
    MAKE_VARIANTS_VE_OUTPOSR( \
            ve, \
            DEFAULT_DELTA_DYNAMIC_VBP_F(ve), \
            inDataLBw, \
            inDataRBw \
    ), \
    MAKE_VARIANTS_VE_OUTPOSR( \
            ve, \
            DEFAULT_FOR_DYNAMIC_VBP_F(ve), \
            inDataLBw, \
            inDataRBw \
    )
#else // uncompressed only
#define MAKE_VARIANTS_VE(ve, outPosRBw, inDataLBw, inDataRBw) \
    MAKE_VARIANT( \
            ve, \
            uncompr_f, \
            uncompr_f, \
            uncompr_f \
    )
#endif

template<class t_varex_t, unsigned t_OutPosRBw, unsigned t_InDataLBw, unsigned t_InDataRBw>
std::vector<typename t_varex_t::variant_t> make_variants() {
    return {
        // Compressed variants.
        MAKE_VARIANTS_VE(scalar<v64<uint64_t>>, t_OutPosRBw, t_InDataLBw, t_InDataRBw),
#ifdef SSE
        MAKE_VARIANTS_VE(sse<v128<uint64_t>>, t_OutPosRBw, t_InDataLBw, t_InDataRBw),
#endif
#ifdef AVXTWO
        MAKE_VARIANTS_VE(avx2<v256<uint64_t>>, t_OutPosRBw, t_InDataLBw, t_InDataRBw),
#endif
#ifdef AVX512
        MAKE_VARIANTS_VE(avx512<v512<uint64_t>>, t_OutPosRBw, t_InDataLBw, t_InDataRBw),
#endif
    };
}

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 2, size_t>::type
        ::for_variant_params<std::string, std::string, std::string, std::string>
        ::for_setting_params<unsigned>;
    varex_t varex(
            {"outCountEstimate"},
            {"vector_extension", "out_pos_r_f", "in_data_l_f", "in_data_r_f"},
            {"datasetIdx"}
    );
    
    const size_t inDataRCountFixed = 16 * 1024 * 1024;
    const unsigned outPosRMaxBwFixed = effective_bitwidth(inDataRCountFixed - 1);
    
    const uint64_t keyCountFixed = 1024;
    const unsigned inDataMaxBwFixed = effective_bitwidth(keyCountFixed - 1);
    
    double selectivity;
    
    unsigned datasetIdx = 0;
    for(auto params : {
        std::make_tuple(0.9),
        std::make_tuple(0.5),
        std::make_tuple(0.1),
    }) {
        datasetIdx++;

        std::tie(selectivity) = params;
        size_t inDataLCount = static_cast<size_t>(keyCountFixed * selectivity);

        varex.print_datagen_started();
        auto inDataLCol = generate_sorted_unique_extraction(
                inDataLCount, keyCountFixed
        );
        auto inDataRCol = generate_with_distr(
                inDataRCountFixed,
                std::uniform_int_distribution<uint64_t>(0, keyCountFixed - 1),
                false
        );
        varex.print_datagen_done();

        std::vector<varex_t::variant_t> variants = make_variants<
                varex_t, outPosRMaxBwFixed, inDataMaxBwFixed, inDataMaxBwFixed
        >();

        varex.execute_variants(variants, datasetIdx, inDataLCol, inDataRCol, 0);

        delete inDataLCol;
        delete inDataRCol;
    }
    
    varex.done();
    
    return 0;
}