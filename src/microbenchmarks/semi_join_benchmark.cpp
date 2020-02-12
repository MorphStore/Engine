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

#define VG_BEGIN \
    if(false) {/*dummy*/}
#define VG_CASE(_outPosMaxVal, _inDataLKeyCount, _keyOffset, _keyFactor) \
    else if( \
            outPosMaxVal == _outPosMaxVal && \
            inDataLKeyCount == _inDataLKeyCount && \
            keyOffset == _keyOffset && \
            keyFactor == _keyFactor \
    ) \
        variants = make_variants< \
                varex_t, \
                effective_bitwidth(_outPosMaxVal), \
                effective_bitwidth( \
                        _keyOffset + (_inDataLKeyCount - 1) * _keyFactor \
                ), \
                effective_bitwidth( \
                        _keyOffset + (_inDataLKeyCount - 1) * _keyFactor \
                ) \
        >();
#define VG_END \
    else throw std::runtime_error( \
            "unexpected combination: outPosMaxVal=" + \
            std::to_string(outPosMaxVal) + ", inDataLKeyCount=" + \
            std::to_string(inDataLKeyCount) \
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
    
    using varex_t = variant_executor_helper<1, 2, size_t>::type
        ::for_variant_params<std::string, std::string, std::string, std::string>
        ::for_setting_params<unsigned>;
    varex_t varex(
            {"outCountEstimate"},
            {"vector_extension", "out_pos_r_f", "in_data_l_f", "in_data_r_f"},
            {"datasetIdx"}
    );
    
    // ========================================================================
    // Specification of the settings.
    // ========================================================================
    
    const size_t inDataRCount1 = 128 * 1024 * 1024;
    const size_t inDataLKeyCount1 = 1024;
    const size_t inDataLKeyCount2 = 1024 * 1024;
    
    const uint64_t _0 = 0;
    const uint64_t _1 = 1;
    const uint64_t min63bit = bitwidth_min<uint64_t>(63);
    
    std::vector<
            std::tuple<size_t, double, size_t, bool, bool, uint64_t, uint64_t>
    > params;
    for(size_t inDataLKeyCount : {
        inDataLKeyCount1,
        inDataLKeyCount2,
    })
        for(double inDataLSel : {
            0.01,
            0.9,
        })
            for(bool inDataRSorted : {
                false,
//                true,
            })
                for(auto keyAffineTransf : {
                    std::make_tuple(_0, _1),
                    std::make_tuple(min63bit, _1),
                    std::make_tuple(
                            min63bit, (min63bit - 1) / (inDataLKeyCount - 1)
                    ),
                }) {
                    uint64_t keyOffset;
                    uint64_t keyFactor;
                    std::tie(keyOffset, keyFactor) = keyAffineTransf;
                    params.push_back(std::make_tuple(
                            inDataLKeyCount, inDataLSel,
                            inDataRCount1, inDataRSorted, false,
                            keyOffset, keyFactor
                    ));
                }
                
    // ========================================================================
    // Variant execution for each setting.
    // ========================================================================
    
    size_t inDataLKeyCount;
    double inDataLSel;
    size_t inDataRCount;
    bool inDataRSorted;
    bool inDataRUpperHalf;
    uint64_t keyOffset;
    uint64_t keyFactor;
    
    unsigned datasetIdx = 0;
    for(auto param : params) {
        datasetIdx++;

        std::tie(
                inDataLKeyCount, inDataLSel,
                inDataRCount, inDataRSorted, inDataRUpperHalf,
                keyOffset, keyFactor
        ) = param;
        const size_t inDataLCount = static_cast<size_t>(
                inDataLKeyCount * inDataLSel
        );
        const size_t outPosMaxVal = inDataRCount - 1;
        
        // --------------------------------------------------------------------
        // Data generation.
        // --------------------------------------------------------------------

        varex.print_datagen_started();
        auto inDataLCol = generate_sorted_unique_extraction(
                inDataLCount, inDataLKeyCount
        );
        auto inDataRCol = generate_with_distr(
                inDataRCount,
                std::uniform_int_distribution<uint64_t>(
                        inDataRUpperHalf ? (inDataLKeyCount / 2) : 0,
                        inDataLKeyCount - 1
                ),
                inDataRSorted
        );
        if(keyOffset != 0 || keyFactor != 1) {
            uint64_t * inDataLData = inDataLCol->get_data();
            for(size_t i = 0; i < inDataLCount; i++)
                inDataLData[i] = keyOffset + inDataLData[i] * keyFactor;
            uint64_t * inDataRData = inDataRCol->get_data();
            for(size_t i = 0; i < inDataRCount; i++)
                inDataRData[i] = keyOffset + inDataRData[i] * keyFactor;
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
        VG_CASE(inDataRCount1 - 1, inDataLKeyCount1, _0, _1)
        VG_CASE(inDataRCount1 - 1, inDataLKeyCount1, min63bit, _1)
        VG_CASE(inDataRCount1 - 1, inDataLKeyCount1, min63bit, (min63bit - 1) / (inDataLKeyCount1 - 1))
        VG_CASE(inDataRCount1 - 1, inDataLKeyCount2, _0, _1)
        VG_CASE(inDataRCount1 - 1, inDataLKeyCount2, min63bit, _1)
        VG_CASE(inDataRCount1 - 1, inDataLKeyCount2, min63bit, (min63bit - 1) / (inDataLKeyCount2 - 1))
        VG_END
        
        // --------------------------------------------------------------------
        // Variant execution.
        // --------------------------------------------------------------------

        varex.execute_variants(variants, datasetIdx, inDataLCol, inDataRCol, 0);

        delete inDataLCol;
        delete inDataRCol;
    }
    
    varex.done();
    
    return 0;
}