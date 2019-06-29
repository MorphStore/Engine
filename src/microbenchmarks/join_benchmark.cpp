/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
//#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/dynamic_vbp.h>

#include <core/utils/variant_executor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <vector/primitives/io.h>
#include <core/utils/preprocessor.h>

//#include <core/operators/general_vectorized/join_uncompr.h>
#include <core/operators/general_vectorized/join.h>




#define MAKE_VARIANT_SJ(ps, format_in1, format_in2, format_out) \
{ \
    new varex_t_sj::operator_wrapper::for_output_formats<format_out>::for_input_formats<format_in1, format_in2>( \
            &semi_join<ps, format_out, format_in1, format_in2> \
    ), \
    STR_EVAL_MACROS(ps), \
    STR_EVAL_MACROS(format_in1), \
    STR_EVAL_MACROS(format_in2), \
    STR_EVAL_MACROS(format_out) \
}

#define MAKE_VARIANT_EJ(ps, format_in1, format_in2, format_out1, format_out2) \
{ \
    new varex_t_ej::operator_wrapper::for_output_formats<format_out1, format_out2>::for_input_formats<format_in1, format_in2>( \
            &join< ps, format_out1, format_out2, format_in1, format_in2> \
    ), \
    STR_EVAL_MACROS(ps), \
    STR_EVAL_MACROS(format_in1), \
    STR_EVAL_MACROS(format_in2), \
    STR_EVAL_MACROS(format_out1), \
    STR_EVAL_MACROS(format_out2) \
}


int main( void ) {
    
    using namespace morphstore;
    using namespace vector;
   
    //using varex_t = variant_executor_helper<1, 2, const size_t>::type
    using varex_t_sj = variant_executor_helper<1, 2, size_t>::type
        ::for_variant_params<std::string, std::string, std::string, std::string>
        ::for_setting_params<size_t,size_t,int>;
    
    varex_t_sj varex_sj(
        {"out-estimate"}, // names of the operator's additional parameters
        {"ps","informat1","informat2","outformat1"}, // names of the variant parameters
        {"inCountLeft", "inCountRight", "match_every_x_values"} // names of the setting parameters
    );

    //using varex_t = variant_executor_helper<1, 2, const size_t>::type
    using varex_t_ej = variant_executor_helper<2, 2, size_t>::type
        ::for_variant_params<std::string, std::string, std::string, std::string, std::string>
        ::for_setting_params<size_t,size_t,int>;
    
    
    varex_t_ej varex_ej(
        {"out-estimate"}, // names of the operator's additional parameters
        {"ps","informat1","informat2","outformat1","outformat2"}, // names of the variant parameters
        {"inCountLeft", "inCountRight", "match_every_x_values"} // names of the setting parameters
    );
    
   
    

    const std::vector<varex_t_ej::variant_t> variants_ej = {
        MAKE_VARIANT_EJ(scalar<v64<uint64_t>>,uncompr_f,uncompr_f,uncompr_f,uncompr_f),
        MAKE_VARIANT_EJ(scalar<v64<uint64_t>>,SINGLE_ARG(dynamic_vbp_f<64,8,1>),SINGLE_ARG(dynamic_vbp_f<64,8,1>),uncompr_f,uncompr_f),
        MAKE_VARIANT_EJ(sse<v128<uint64_t>>,SINGLE_ARG(dynamic_vbp_f<128,16,2>),SINGLE_ARG(dynamic_vbp_f<128,16,2>),uncompr_f,uncompr_f),
        #ifdef AVXTWO
        MAKE_VARIANT_EJ(avx2<v256<uint64_t>>,SINGLE_ARG(dynamic_vbp_f<256,32,4>),SINGLE_ARG(dynamic_vbp_f<256,32,4>),uncompr_f,uncompr_f),
        #endif
        #ifdef AVX512
        MAKE_VARIANT_EJ(avx512<v512<uint64_t>>,SINGLE_ARG(dynamic_vbp_f<512,64,8>),SINGLE_ARG(dynamic_vbp_f<512,64,8>),uncompr_f,uncompr_f)
        #endif
        
    }; 


    const std::vector<varex_t_sj::variant_t> variants_sj = {
        MAKE_VARIANT_SJ(scalar<v64<uint64_t>>,uncompr_f,uncompr_f,uncompr_f),
        MAKE_VARIANT_SJ(scalar<v64<uint64_t>>,SINGLE_ARG(dynamic_vbp_f<64,8,1>),SINGLE_ARG(dynamic_vbp_f<64,8,1>),uncompr_f),
        MAKE_VARIANT_SJ(sse<v128<uint64_t>>,SINGLE_ARG(dynamic_vbp_f<128,16,2>),SINGLE_ARG(dynamic_vbp_f<128,16,2>),uncompr_f),
        #ifdef AVXTWO
        MAKE_VARIANT_SJ(avx2<v256<uint64_t>>,SINGLE_ARG(dynamic_vbp_f<256,32,4>),SINGLE_ARG(dynamic_vbp_f<256,32,4>),uncompr_f),
        #endif
        #ifdef AVX512
        MAKE_VARIANT_SJ(avx512<v512<uint64_t>>,SINGLE_ARG(dynamic_vbp_f<512,64,8>),SINGLE_ARG(dynamic_vbp_f<512,64,8>),uncompr_f)
        #endif
    };
       
    

    
    
    // Define the setting parameters for semi join
    const std::vector<varex_t_sj::setting_t> settingParams_sj = {
        // left count, right count, match every x values
        {1000, 1000, 1},
        {1000, 1000, 10},
        {1000, 1000, 100}
    };
    
    // Define the setting parameters for equi join.
    const std::vector<varex_t_ej::setting_t> settingParams_ej = {
        // left count, right count, match every x values
        {1000, 100, 1},
        {1000, 100, 10},
        {1000, 100, 100}
    };
    
    
    
      
    std::cout << "Equi-Join\n";
    for(const varex_t_ej::setting_t sp : settingParams_ej) {
        // Extract the individual setting parameters.
        size_t inDataCountLeft;
        size_t inDataCountRight;
        int matches;
        std::tie(inDataCountLeft, inDataCountRight, matches) = sp;
        
        // Generate the data.
        varex_ej.print_datagen_started();
        auto inDataColLeft = generate_sorted_unique(inDataCountLeft, 1, 1);
        
        auto inDataColRight = generate_sorted_unique(inDataCountRight, 1, matches);
        
        varex_ej.print_datagen_done();
        
        
        
        // Execute the variants.
        varex_ej.execute_variants(
            // Variants to execute
            variants_ej,
            // Setting parameters
            inDataCountLeft, inDataCountRight, matches,
            // Input columns / setting
            inDataColLeft, inDataColRight, 0
        );   
        
        delete inDataColLeft;
        delete inDataColRight;
    }
    
    varex_ej.done();
   
    std::cout << "Semi-Join\n";
    for(const varex_t_sj::setting_t sp : settingParams_sj) {
        // Extract the individual setting parameters.
        size_t inDataCountLeft;
        size_t inDataCountRight;
        int matches;
        std::tie(inDataCountLeft, inDataCountRight, matches) = sp;
        
        // Generate the data.
        varex_sj.print_datagen_started();
        auto inDataColLeft = generate_sorted_unique(inDataCountLeft, 1, 1);
        
        auto inDataColRight = generate_sorted_unique(inDataCountRight, 1, matches);
        
        varex_sj.print_datagen_done();
        
     
        // Execute the variants.
        varex_sj.execute_variants(
            // Variants to execute
            variants_sj,
            // Setting parameters
            inDataCountLeft, inDataCountRight, matches,
            // Input columns / setting
            inDataColLeft, inDataColRight, 0
        );    
        
        delete inDataColLeft;
        delete inDataColRight;
     }
    
    varex_sj.done();  
    
    
}