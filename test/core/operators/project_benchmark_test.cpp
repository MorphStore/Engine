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
 * @file project_benchmark_test.cpp
 * @brief A testing and micro-benchmarking suite for multiple/all variants of
 * the project-operator.
 */

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/vbp.h>
#include <core/operators/scalar/project_compr.h>
#include <core/operators/scalar/project_uncompr.h>
#include <core/operators/vectorized/project_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <core/utils/variant_executor.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <iostream>
#include <random>
#include <tuple>
#include <map>
#include <vector>

using namespace morphstore;
using namespace vectorlib;

#define MAKE_VARIANT(ve, in_pos_f) \
{ \
    new ve_t::operator_wrapper \
        ::for_output_formats<uncompr_f> \
        ::for_input_formats<uncompr_f, in_pos_f>( \
            &project< \
                    ve, \
                    uncompr_f, \
                    uncompr_f, in_pos_f \
            > \
    ), \
    STR_EVAL_MACROS(ve), \
    STR_EVAL_MACROS(in_pos_f) \
}
    
// This must be a macro, because if it is a C++ variable/constant, then the
// output will always contain the identifier/expression, but not the value.
// The value must be a literal here, it can be calculated as
// sizeof(__m128i) / sizeof(uint64_t) .
#define STEP_128 2

int main(void) {
#ifdef MSV_NO_SELFMANAGED_MEMORY
    // Setup.
    using ve_t = variant_executor_helper<1, 2>::type
            ::for_variant_params<std::string, std::string>
            ::for_setting_params<size_t, size_t>;
    ve_t ve(
            {},
            {"ps", "in_pos_f"},
            {"inDataCount", "inPosCount"}
    );
    
    // These variants can be executed for all input columns.
    const std::vector<ve_t::variant_t> variants = {
        MAKE_VARIANT(scalar<v64<uint64_t>>, uncompr_f),
        MAKE_VARIANT(sse<v128<uint64_t>>, uncompr_f),
        MAKE_VARIANT(avx2<v256<uint64_t>>, uncompr_f)
    };
    
    // These variants can only be executed if their static bit width is high
    // enough to represent the highest possible position.
    std::map<unsigned, std::vector<ve_t::variant_t> > staticVPBVariantsByBw = {
        // Generated with Python:
        // for bw in range(1, 64+1):
        //   print("{{{: >2}, {{MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<{: >2}, STEP_128>>))}}}}{}".format(bw, bw, "," if bw < 64 else ""))
//        { 1, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l< 1, STEP_128>>))}},
//        { 2, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l< 2, STEP_128>>))}},
//        { 3, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l< 3, STEP_128>>))}},
//        { 4, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l< 4, STEP_128>>))}},
//        { 5, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l< 5, STEP_128>>))}},
//        { 6, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l< 6, STEP_128>>))}},
//        { 7, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l< 7, STEP_128>>))}},
//        { 8, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l< 8, STEP_128>>))}},
//        { 9, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l< 9, STEP_128>>))}},
//        {10, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<10, STEP_128>>))}},
//        {11, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<11, STEP_128>>))}},
//        {12, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<12, STEP_128>>))}},
//        {13, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<13, STEP_128>>))}},
//        {14, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<14, STEP_128>>))}},
//        {15, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<15, STEP_128>>))}},
//        {16, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<16, STEP_128>>))}},
//        {17, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<17, STEP_128>>))}},
//        {18, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<18, STEP_128>>))}},
//        {19, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<19, STEP_128>>))}},
//        {20, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<20, STEP_128>>))}},
//        {21, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<21, STEP_128>>))}},
//        {22, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<22, STEP_128>>))}},
//        {23, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<23, STEP_128>>))}},
//        {24, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<24, STEP_128>>))}},
//        {25, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<25, STEP_128>>))}},
//        {26, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<26, STEP_128>>))}},
//        {27, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<27, STEP_128>>))}},
//        {28, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<28, STEP_128>>))}},
//        {29, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<29, STEP_128>>))}},
//        {30, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<30, STEP_128>>))}},
//        {31, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<31, STEP_128>>))}},
//        {32, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<32, STEP_128>>))}},
//        {33, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<33, STEP_128>>))}},
//        {34, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<34, STEP_128>>))}},
//        {35, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<35, STEP_128>>))}},
//        {36, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<36, STEP_128>>))}},
//        {37, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<37, STEP_128>>))}},
//        {38, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<38, STEP_128>>))}},
//        {39, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<39, STEP_128>>))}},
//        {40, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<40, STEP_128>>))}},
//        {41, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<41, STEP_128>>))}},
//        {42, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<42, STEP_128>>))}},
//        {43, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<43, STEP_128>>))}},
//        {44, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<44, STEP_128>>))}},
//        {45, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<45, STEP_128>>))}},
//        {46, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<46, STEP_128>>))}},
//        {47, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<47, STEP_128>>))}},
//        {48, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<48, STEP_128>>))}},
//        {49, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<49, STEP_128>>))}},
//        {50, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<50, STEP_128>>))}},
//        {51, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<51, STEP_128>>))}},
//        {52, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<52, STEP_128>>))}},
//        {53, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<53, STEP_128>>))}},
//        {54, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<54, STEP_128>>))}},
//        {55, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<55, STEP_128>>))}},
//        {56, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<56, STEP_128>>))}},
//        {57, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<57, STEP_128>>))}},
//        {58, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<58, STEP_128>>))}},
//        {59, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<59, STEP_128>>))}},
//        {60, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<60, STEP_128>>))}},
//        {61, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<61, STEP_128>>))}},
//        {62, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<62, STEP_128>>))}},
//        {63, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<63, STEP_128>>))}},
//        {64, {MAKE_VARIANT(scalar, SINGLE_ARG(static_vbp_f<vbp_l<64, STEP_128>>))}}
    };
    
    // Variant execution for several settings.
    for(const std::tuple<size_t, size_t> settingParams : {
        // Setting in which a few positions refer to a large data column.
        std::make_tuple(16 * 1024, 128 * 1000 * 1000),
        // Setting in which sequentially reading the input position column
        // should have a significant impact on the runtime.
        std::make_tuple(2, 128 * 1000 * 1000),
        // Setting in which sequentially reading the input position column
        // should not have a significant impact on the runtime.
        std::make_tuple(1000 * 1000, 128 * 1000)
    }) {
        size_t inDataCount;
        size_t inPosCount;
        std::tie(inDataCount, inPosCount) = settingParams;

        ve.print_datagen_started();
        auto inDataCol = ColumnGenerator::generate_with_distr(
                inDataCount,
                std::uniform_int_distribution<uint64_t>(100, 200),
                false
        );
        auto inPosCol = ColumnGenerator::generate_with_distr(
                inPosCount,
                std::uniform_int_distribution<uint64_t>(0, inDataCount - 1),
                false
        );
        ve.print_datagen_done();
        
        std::vector<ve_t::variant_t> variantsToUse(variants);
        // Add all operator variants using the static_vbp format with a high
        // enough bit width.
        const unsigned minBw = effective_bitwidth(inDataCount - 1);
        for(const auto & vs : staticVPBVariantsByBw)
            if(vs.first >= minBw)
                for(const auto & v : vs.second)
                    variantsToUse.push_back(v);
        
        ve.execute_variants(
                // Variants to execute
                variantsToUse,
                // Setting parameters
                inDataCount, inPosCount,
                // Input columns
                inDataCol, inPosCol
        );
        
        delete inPosCol;
        delete inDataCol;
    }
    
    // Finish and print a summary.
    ve.done();
    
    return !ve.good();
#else
    // @todo Make it work with self-managed memory.
    std::cerr
            << "Currently, this benchmark only works with non-self-managed "
            << "memory. Compile MorphStore with build.sh -noSelfManaging ."
            << std::endl;
    return 1;
#endif
}
