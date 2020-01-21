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

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 2>::type
        ::for_variant_params<std::string, std::string, std::string, std::string>
        ::for_setting_params<unsigned>;
    varex_t varex(
            {},
            {"vector_extension", "out_data_f", "in_data_f", "in_pos_f"},
            {"datasetIdx"}
    );
    
    const size_t inDataCountFixed = 16 * 1024 * 1024;
    const unsigned inPosMaxBwFixed = effective_bitwidth(inDataCountFixed - 1);
    
    const uint64_t inDataMax1 = 200;
    const uint64_t inDataMax2 = 300;
    const uint64_t inDataMax3 = 500;
    const unsigned inDataMaxBw1 = effective_bitwidth(inDataMax1);
    const unsigned inDataMaxBw2 = effective_bitwidth(inDataMax2);
    const unsigned inDataMaxBw3 = effective_bitwidth(inDataMax3);
    
    uint64_t inDataMax;
    double selectivity;
    
    unsigned datasetIdx = 0;
    for(auto params : {
        std::make_tuple(inDataMax1, 0.9),
        std::make_tuple(inDataMax2, 0.5),
        std::make_tuple(inDataMax3, 0.1),
    }) {
        datasetIdx++;

        std::tie(inDataMax, selectivity) = params;
        size_t inPosCount = static_cast<size_t>(inDataCountFixed * selectivity);
        
        const unsigned inDataMaxBw = effective_bitwidth(inDataMax);

        varex.print_datagen_started();
        auto inDataCol = generate_with_distr(
                inDataCountFixed,
                std::uniform_int_distribution<uint64_t>(0, inDataMax),
                false
        );
        auto inPosCol = generate_with_distr(
                inPosCount,
                std::uniform_int_distribution<uint64_t>(0, inDataCountFixed - 1),
                true
        );
        varex.print_datagen_done();

        std::vector<varex_t::variant_t> variants;
        
        // Only enumerate the maximum bit widths that might actually be
        // encountered depending on the parameters of the data generation.
        // We do not need all 64 bit widths. This greatly reduces the
        // compilation time.
#if 0
        switch(inDataMaxBw) {
            case inDataMaxBw1: variants = make_variants<varex_t, inDataMaxBw1, inDataMaxBw1, inPosMaxBwFixed>(); break;
            case inDataMaxBw2: variants = make_variants<varex_t, inDataMaxBw2, inDataMaxBw2, inPosMaxBwFixed>(); break;
            case inDataMaxBw3: variants = make_variants<varex_t, inDataMaxBw3, inDataMaxBw3, inPosMaxBwFixed>(); break;
            default:
                throw std::runtime_error(
                        "unsupported inDataMaxBw: " +
                        std::to_string(inDataMaxBw)
                );
        }
#else
        // With if-else we do not need to think about duplicate case labels.
             if(inDataMaxBw == inDataMaxBw1) variants = make_variants<varex_t, inDataMaxBw1, inDataMaxBw1, inPosMaxBwFixed>();
        else if(inDataMaxBw == inDataMaxBw2) variants = make_variants<varex_t, inDataMaxBw2, inDataMaxBw2, inPosMaxBwFixed>();
        else if(inDataMaxBw == inDataMaxBw3) variants = make_variants<varex_t, inDataMaxBw3, inDataMaxBw3, inPosMaxBwFixed>();
        else throw std::runtime_error(
                "unsupported inDataMaxBw: " + std::to_string(inDataMaxBw)
        );
#endif

        varex.execute_variants(variants, datasetIdx, inDataCol, inPosCol);

        delete inDataCol;
        delete inPosCol;
    }
    
    varex.done();
    
    return 0;
}