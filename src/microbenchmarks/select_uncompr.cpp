/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/static_vbp.h>


#include <vector/vector_extension_structs.h>
#include <vector/primitives/io.h>
#include <core/utils/preprocessor.h>

#include <core/operators/general_vectorized/select_uncompr.h>

#include <core/utils/variant_executor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>



#define MAKE_VARIANT(fn,ps) \
{ \
    new varex_t::operator_wrapper::for_output_formats<uncompr_f>::for_input_formats<uncompr_f>( \
            &morphstore::select<fn,ps, uncompr_f, uncompr_f> \
    ), \
    STR_EVAL_MACROS(ps), \
    STR_EVAL_MACROS(fn) \
}

int main( void ) {
    
    using namespace morphstore;
    using namespace vectorlib;
   
    using varex_t = variant_executor_helper<1, 1, uint64_t, size_t>::type
        ::for_variant_params<std::string,std::string>
        ::for_setting_params<size_t, float>;
    
    varex_t varex(
        {"predicate","estimate"}, // names of the operator's additional parameters
        {"fn","ps"}, // names of the variant parameters
        {"inDataCount","selectivity"} // names of the setting parameters
    );


    
    //The following are the variants for the general_vectorized version
    const std::vector<varex_t::variant_t> variants = {
        MAKE_VARIANT(equal,scalar<v64<uint64_t>>),
        #ifdef SSE
        MAKE_VARIANT(equal,sse<v128<uint64_t>>),
        #endif
        #ifdef AVXTWO
        MAKE_VARIANT(equal,avx2<v256<uint64_t>>),
        #endif
        #ifdef AVX512
        MAKE_VARIANT(equal,avx512<v512<uint64_t>>)
        #endif
    };
    
    // Define the setting parameters.
    const std::vector<varex_t::setting_t> settingParams = {
        //size, selectivity
        {82705, 0.1},
        {82705, 0.5},
        {82705, 0.9}
    };
    
    for(const varex_t::setting_t sp : settingParams) {
        // Extract the individual setting parameters.
        size_t inDataCount;
        float selectivity;
        std::tie(inDataCount, selectivity) = sp;
                varex.print_datagen_started();
                const size_t countMatches = static_cast<size_t>(
                        static_cast<float>(inDataCount) * selectivity
                );
                auto inDataCol = generate_exact_number(
                        inDataCount,
                        countMatches,
                        0,
                        bitwidth_max<uint64_t>(64)
                );
                varex.print_datagen_done();
        
        // Execute the variants.
        varex.execute_variants(
            // Variants to execute
            variants,
            // Setting parameters
            inDataCount, selectivity,
            // Input columns / setting
            inDataCol, 0, 0
        );
        
        delete inDataCol;
    }
    
    varex.done();
    
}