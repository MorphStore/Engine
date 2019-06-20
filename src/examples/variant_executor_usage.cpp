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
 * @file variant_executor_usage.cpp
 * @brief This file illustrates the usage of variant_executor for comparing
 * different variants of one operator.
 * There is a tutorial on that in the documentation, which refers to this file.
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/operators/scalar/project_uncompr.h>
#include <core/operators/vectorized/project_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/preprocessor.h>
#include <core/utils/variant_executor.h>

#include <vector/scalar/extension_scalar.h>
#include <vector/scalar/primitives/calc_scalar.h>
#include <vector/scalar/primitives/create_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/scalar/primitives/logic_scalar.h>
#ifdef AVXTWO
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/logic_avx2.h>
#endif
#ifdef AVX512
#include <vector/simd/avx512/extension_avx512.h>
#include <vector/simd/avx512/primitives/calc_avx512.h>
#include <vector/simd/avx512/primitives/create_avx512.h>
#include <vector/simd/avx512/primitives/io_avx512.h>
#include <vector/simd/avx512/primitives/logic_avx512.h>
#endif
#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/logic_sse.h>

#include <iostream>
#include <random>
#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vector;

// A macro expanding to an initializer list for a variant.
#define MAKE_VARIANT(ve) \
{ \
    new varex_t::operator_wrapper \
        ::for_output_formats<uncompr_f> \
        ::for_input_formats<uncompr_f, uncompr_f>( \
            &project< \
                    ve, \
                    uncompr_f, \
                    uncompr_f, uncompr_f \
            > \
    ), \
    STR_EVAL_MACROS(ve) \
}
    
int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    // Create an instance of a variant executor.
    using varex_t = variant_executor_helper<1, 2>::type
            ::for_variant_params<std::string>
            ::for_setting_params<size_t, size_t>;
    varex_t varex(
            {}, // names of the operator's additional parameters
            {"ve"}, // names of the variant parameters
            {"inDataCount", "inPosCount"} // names of the setting parameters
    );
    
    // Define the variants.
    const std::vector<varex_t::variant_t> variants = {
        MAKE_VARIANT(scalar<v64<uint64_t>>),
        MAKE_VARIANT(sse<v128<uint64_t>>),
#ifdef AVXTWO
        MAKE_VARIANT(avx2<v256<uint64_t>>),
#endif
    };
    
    // Define the setting parameters.
    const std::vector<varex_t::setting_t> settingParams = {
        // inDataCount, inPosCount
        {100, 1000},
        {123, 1234}
    };
    
    // Variant execution for several settings.
    for(const varex_t::setting_t sp : settingParams) {
        // Extract the individual setting parameters.
        size_t inDataCount;
        size_t inPosCount;
        std::tie(inDataCount, inPosCount) = sp;

        // Generate the data.
        varex.print_datagen_started();
        auto inDataCol = generate_with_distr(
                inDataCount,
                std::uniform_int_distribution<uint64_t>(100, 200),
                false
        );
        auto inPosCol = generate_with_distr(
                inPosCount,
                std::uniform_int_distribution<uint64_t>(0, inDataCount - 1),
                false
        );
        varex.print_datagen_done();
        
        // Execute the variants.
        varex.execute_variants(
                // Variants to execute
                variants,
                // Setting parameters
                inDataCount, inPosCount,
                // Input columns / setting
                inDataCol, inPosCol
        );
        
        // Delete the generated data.
        delete inPosCol;
        delete inDataCol;
    }
    
    // Finish and print a summary.
    varex.done();
    
    return !varex.good();
}