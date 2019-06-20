/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <core/memory/mm_glob.h>

//#include "../../core/operators/operator_test_frames.h"
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>

#include <vector/simd/avx512/extension_avx512.h>
#include <vector/simd/avx512/primitives/calc_avx512.h>
#include <vector/simd/avx512/primitives/io_avx512.h>
#include <vector/simd/avx512/primitives/create_avx512.h>

#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>



#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>

#include <core/operators/general_vectorized/project_uncompr.h>

#include <core/utils/variant_executor.h>

#define TEST_DATA_COUNT 100

#define MAKE_VARIANT(ps) \
{ \
    new varex_t::operator_wrapper::for_output_formats<uncompr_f>::for_input_formats<uncompr_f, uncompr_f>( \
            &project<ps, uncompr_f, uncompr_f, uncompr_f> \
    ), \
    STR_EVAL_MACROS(ps) \
}

int main( void ) {
    
    using namespace morphstore;
    using namespace vector;
   
    using varex_t = variant_executor_helper<1, 2>::type
        ::for_variant_params<std::string>
        ::for_setting_params<size_t,size_t>;
    
    varex_t varex(
        {}, // names of the operator's additional parameters
        {"ps"}, // names of the variant parameters
        {"inDataCount", "inPosCount"} // names of the setting parameters
    );

    // This is a std::tuple.
    varex_t::variant_t myVariant = {
            // Wrapper for the function pointer.
             new varex_t::operator_wrapper::for_output_formats<uncompr_f>::for_input_formats<uncompr_f, uncompr_f>(
                    // Function pointer to the actual operator function.
                    &project<scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f> 
            ),
            // The variant key.
            "scalar"
    };
    
    const std::vector<varex_t::variant_t> variants = {
        MAKE_VARIANT(scalar<v64<uint64_t>>),
        MAKE_VARIANT(sse<v128<uint64_t>>),
        MAKE_VARIANT(avx2<v256<uint64_t>>),
        MAKE_VARIANT(avx512<v512<uint64_t>>)
    };
    
    // Define the setting parameters.
    const std::vector<varex_t::setting_t> settingParams = {
        // inDataCount, inPosCount
        {100000, 100000},//<1MB
        {146309, 146309},//Primzahl
        {10000000, 10000000},//Doesn't fit into cache
    };
    
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
    }
    
    varex.done();
    varex.good();
}