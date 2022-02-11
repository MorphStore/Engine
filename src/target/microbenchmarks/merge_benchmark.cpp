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
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/static_vbp.h>


#include <vector/vector_extension_structs.h>
#include <vector/primitives/io.h>
#include <core/utils/preprocessor.h>

#include <core/operators/scalar/merge_uncompr.h>
#include <core/operators/vectorized/merge_uncompr.h>

#include <core/utils/variant_executor.h>

#ifdef AVX512
#include <vector/simd/avx512/extension_avx512.h>
#include <vector/simd/avx512/primitives/calc_avx512.h>
#include <vector/simd/avx512/primitives/compare_avx512.h>
#include <vector/simd/avx512/primitives/io_avx512.h>
#include <vector/simd/avx512/primitives/create_avx512.h>
#include <vector/simd/avx512/primitives/logic_avx512.h>
#include <vector/simd/avx512/primitives/manipulate_avx512.h>
#include <vector/simd/avx512/primitives/extract_avx512.h>
#endif

#ifdef AVXTWO
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/logic_avx2.h>
#include <vector/simd/avx2/primitives/manipulate_avx2.h>
#include <vector/simd/avx2/primitives/extract_avx2.h>
#endif

#ifdef SSE
#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/compare_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include <vector/simd/sse/primitives/logic_sse.h>
#include <vector/simd/sse/primitives/manipulate_sse.h>
#include <vector/simd/sse/primitives/extract_sse.h>
#endif

#include <vector/scalar/extension_scalar.h>
#include <vector/scalar/primitives/calc_scalar.h>
#include <vector/scalar/primitives/compare_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/scalar/primitives/create_scalar.h>
#include <vector/scalar/primitives/logic_scalar.h>
#include <vector/scalar/primitives/manipulate_scalar.h>
#include <vector/scalar/primitives/extract_scalar.h>

//#include <core/operators/general_vectorized/merge_uncompr.h>

#include <iostream>
#include <random>
#include <tuple>
#include <map>
#include <vector>


#define MAKE_VARIANT(ps) \
{ \
    new varex_t::operator_wrapper::for_output_formats<uncompr_f>::for_input_formats<uncompr_f, uncompr_f>( \
            &merge_sorted<ps, uncompr_f, uncompr_f, uncompr_f> \
    ), \
    STR_EVAL_MACROS(ps) \
}

int main( void ) {
    
    using namespace morphstore;
    using namespace vectorlib;
   
    using varex_t = variant_executor_helper<1, 2>::type
        ::for_variant_params<std::string>
        ::for_setting_params<size_t,size_t>;
    
    varex_t varex(
        {}, // names of the operator's additional parameters
        {"ps"}, // names of the variant parameters
        {"inDataCount", "inPosCount"} // names of the setting parameters
    );

    //The following are the variants for the hand written versions
    const std::vector<varex_t::variant_t> variants = {
        MAKE_VARIANT(scalar<v64<uint64_t>>),
        #ifdef AVXTWO
        MAKE_VARIANT(avx2<v256<uint64_t>>),
        #endif
        
    };
    
    //The following are the variants for the general_vectorized version
   /* const std::vector<varex_t::variant_t> variants = {
        MAKE_VARIANT(scalar<v64<uint64_t>>),
        MAKE_VARIANT(sse<v128<uint64_t>>),
        #ifdef AVXTWO
        MAKE_VARIANT(avx2<v256<uint64_t>>),
        #endif
        #ifdef AVX512
        MAKE_VARIANT(avx512<v512<uint64_t>>)
        #endif
    };*/
    
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
        auto inDataCol = ColumnGenerator::generate_with_distr(
            inDataCount,
            std::uniform_int_distribution<uint64_t>(100, 200),
                true
        );
        auto inPosCol = ColumnGenerator::generate_with_distr(
            inPosCount,
            std::uniform_int_distribution<uint64_t>(0, inDataCount - 1),
                true
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
    
}
