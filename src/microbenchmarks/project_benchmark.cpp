/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <core/memory/noselfmanaging_helper.h>
#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/static_vbp.h>


#include <vector/vector_extension_structs.h>
#include <vector/primitives/io.h>
#include <core/utils/preprocessor.h>
//#include <core/operators/general_vectorized/project_uncompr.h>
#include <core/operators/scalar/project_uncompr.h>
#include <core/operators/general_vectorized/project_compr.h>

#include <core/utils/variant_executor.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <iostream>
#include <random>
#include <tuple>
#include <map>
#include <vector>



#define MAKE_VARIANT_CLASSICAL(ps) \
{ \
    new varex_t::operator_wrapper::for_output_formats<uncompr_f>::for_input_formats<uncompr_f, uncompr_f>( \
            &project<ps, uncompr_f, uncompr_f, uncompr_f> \
    ), \
    STR_EVAL_MACROS(ps), \
    "uncompr_f", \
    "uncompr_f", \
    "uncompr_f" \
}

#define MAKE_VARIANT_WIT(ps, out_data_f, in_data_f, in_pos_f) \
{ \
    new varex_t::operator_wrapper::for_output_formats<out_data_f>::for_input_formats<in_data_f, in_pos_f>( \
            &my_project_wit_t<ps, out_data_f, in_data_f, in_pos_f>::apply \
    ), \
    STR_EVAL_MACROS(ps), \
    STR_EVAL_MACROS(out_data_f), \
    STR_EVAL_MACROS(in_data_f), \
    STR_EVAL_MACROS(in_pos_f) \
}

#ifndef AVXTWO
#define MAKE_VARIANTS(bwData, bwPos) \
    MAKE_VARIANT_CLASSICAL(scalar<v64<uint64_t>>), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, uncompr_f                                 , uncompr_f                                 , uncompr_f), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, uncompr_f                                 , uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 1>>)), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), uncompr_f), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 1>>)), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), uncompr_f                                 , uncompr_f), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 1>>)), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), uncompr_f), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 1>>))
#else
#define MAKE_VARIANTS(bwData, bwPos) \
    MAKE_VARIANT_CLASSICAL(scalar<v64<uint64_t>>), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, uncompr_f                                 , uncompr_f                                 , uncompr_f), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, uncompr_f                                 , uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 1>>)), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), uncompr_f), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 1>>)), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), uncompr_f                                 , uncompr_f), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 1>>)), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), uncompr_f), \
    MAKE_VARIANT_WIT(scalar<v64<uint64_t>>, SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), SINGLE_ARG(static_vbp_f<vbp_l<bwData, 1>>), SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 1>>)), \
    MAKE_VARIANT_WIT(sse<v128<uint64_t>>  , uncompr_f                                 , uncompr_f                                 , uncompr_f), \
    MAKE_VARIANT_WIT(sse<v128<uint64_t>>  , uncompr_f                                 , uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 2>>)), \
    MAKE_VARIANT_WIT(sse<v128<uint64_t>>  , uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 2>>), uncompr_f), \
    MAKE_VARIANT_WIT(sse<v128<uint64_t>>  , uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 2>>), SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 2>>)), \
    MAKE_VARIANT_WIT(sse<v128<uint64_t>>  , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 2>>), uncompr_f                                 , uncompr_f), \
    MAKE_VARIANT_WIT(sse<v128<uint64_t>>  , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 2>>), uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 2>>)), \
    MAKE_VARIANT_WIT(sse<v128<uint64_t>>  , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 2>>), SINGLE_ARG(static_vbp_f<vbp_l<bwData, 2>>), uncompr_f), \
    MAKE_VARIANT_WIT(sse<v128<uint64_t>>  , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 2>>), SINGLE_ARG(static_vbp_f<vbp_l<bwData, 2>>), SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 2>>)), \
    MAKE_VARIANT_WIT(avx2<v256<uint64_t>> , uncompr_f                                 , uncompr_f                                 , uncompr_f), \
    MAKE_VARIANT_WIT(avx2<v256<uint64_t>> , uncompr_f                                 , uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 4>>)), \
    MAKE_VARIANT_WIT(avx2<v256<uint64_t>> , uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 4>>), uncompr_f), \
    MAKE_VARIANT_WIT(avx2<v256<uint64_t>> , uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 4>>), SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 4>>)), \
    MAKE_VARIANT_WIT(avx2<v256<uint64_t>> , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 4>>), uncompr_f                                 , uncompr_f), \
    MAKE_VARIANT_WIT(avx2<v256<uint64_t>> , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 4>>), uncompr_f                                 , SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 4>>)), \
    MAKE_VARIANT_WIT(avx2<v256<uint64_t>> , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 4>>), SINGLE_ARG(static_vbp_f<vbp_l<bwData, 4>>), uncompr_f), \
    MAKE_VARIANT_WIT(avx2<v256<uint64_t>> , SINGLE_ARG(static_vbp_f<vbp_l<bwData, 4>>), SINGLE_ARG(static_vbp_f<vbp_l<bwData, 4>>), SINGLE_ARG(static_vbp_f<vbp_l<bwPos, 4>>))
#endif   

int main( void ) {
    using namespace morphstore;
    using namespace vectorlib;
    
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
   
    using varex_t = variant_executor_helper<1, 2>::type
        ::for_variant_params<std::string, std::string, std::string, std::string>
        ::for_setting_params<size_t,size_t>;
    
    varex_t varex(
        {}, // names of the operator's additional parameters
        {"ps", "out_data_f", "in_data_f", "in_pos_f"}, // names of the variant parameters
        {"inDataCount", "inPosCount"} // names of the setting parameters
    );
    
    // Define the setting parameters.
    const std::vector<varex_t::setting_t> settingParams = {
        // inDataCount, inPosCount
        
        {256*1024, 64*1024},
        {256*1024+17, 64*1024},
        {256*1024, 64*1024+16},
        {256*1024, 64*1024+17},
        {256*1024+47, 64*1024+17},
        
        {100000, 100000},//<1MB
        {146309, 146309},//Primzahl
        {10000000, 10000000},//Doesn't fit into cache
    };
    
    const uint64_t minVal = 100;
    const uint64_t maxVal = 200;
    const unsigned bwData = effective_bitwidth(maxVal);
    
    for(const varex_t::setting_t sp : settingParams) {
        // Extract the individual setting parameters.
        size_t inDataCount;
        size_t inPosCount;
        std::tie(inDataCount, inPosCount) = sp;
        
        // Generate the data.
        varex.print_datagen_started();
        auto inDataCol = generate_with_distr(
            inDataCount,
            std::uniform_int_distribution<uint64_t>(minVal, maxVal),
            false
        );
        auto inPosCol = generate_with_distr(
            inPosCount,
            std::uniform_int_distribution<uint64_t>(0, inDataCount - 1),
            false
        );
        varex.print_datagen_done();
        
        const unsigned bwPos = effective_bitwidth(inDataCount - 1);
    
        for(
                unsigned bw = bwPos;
                bw <= std::numeric_limits<uint64_t>::digits;
                bw++
        ) {
            std::vector<varex_t::variant_t> variants;
            switch(bw) {
                // Generated with python:
                // for bw in range(1, 64+1):
                //   print("            case {: >2}: variants = {{MAKE_VARIANTS(bwData, {: >2})}}; break;".format(bw, bw))
                case  1: variants = {MAKE_VARIANTS(bwData,  1)}; break;
                case  2: variants = {MAKE_VARIANTS(bwData,  2)}; break;
                case  3: variants = {MAKE_VARIANTS(bwData,  3)}; break;
                case  4: variants = {MAKE_VARIANTS(bwData,  4)}; break;
                case  5: variants = {MAKE_VARIANTS(bwData,  5)}; break;
                case  6: variants = {MAKE_VARIANTS(bwData,  6)}; break;
                case  7: variants = {MAKE_VARIANTS(bwData,  7)}; break;
                case  8: variants = {MAKE_VARIANTS(bwData,  8)}; break;
                case  9: variants = {MAKE_VARIANTS(bwData,  9)}; break;
                case 10: variants = {MAKE_VARIANTS(bwData, 10)}; break;
                case 11: variants = {MAKE_VARIANTS(bwData, 11)}; break;
                case 12: variants = {MAKE_VARIANTS(bwData, 12)}; break;
                case 13: variants = {MAKE_VARIANTS(bwData, 13)}; break;
                case 14: variants = {MAKE_VARIANTS(bwData, 14)}; break;
                case 15: variants = {MAKE_VARIANTS(bwData, 15)}; break;
                case 16: variants = {MAKE_VARIANTS(bwData, 16)}; break;
                case 17: variants = {MAKE_VARIANTS(bwData, 17)}; break;
                case 18: variants = {MAKE_VARIANTS(bwData, 18)}; break;
                case 19: variants = {MAKE_VARIANTS(bwData, 19)}; break;
                case 20: variants = {MAKE_VARIANTS(bwData, 20)}; break;
                case 21: variants = {MAKE_VARIANTS(bwData, 21)}; break;
                case 22: variants = {MAKE_VARIANTS(bwData, 22)}; break;
                case 23: variants = {MAKE_VARIANTS(bwData, 23)}; break;
                case 24: variants = {MAKE_VARIANTS(bwData, 24)}; break;
                case 25: variants = {MAKE_VARIANTS(bwData, 25)}; break;
                case 26: variants = {MAKE_VARIANTS(bwData, 26)}; break;
                case 27: variants = {MAKE_VARIANTS(bwData, 27)}; break;
                case 28: variants = {MAKE_VARIANTS(bwData, 28)}; break;
                case 29: variants = {MAKE_VARIANTS(bwData, 29)}; break;
                case 30: variants = {MAKE_VARIANTS(bwData, 30)}; break;
                case 31: variants = {MAKE_VARIANTS(bwData, 31)}; break;
                case 32: variants = {MAKE_VARIANTS(bwData, 32)}; break;
                case 33: variants = {MAKE_VARIANTS(bwData, 33)}; break;
                case 34: variants = {MAKE_VARIANTS(bwData, 34)}; break;
                case 35: variants = {MAKE_VARIANTS(bwData, 35)}; break;
                case 36: variants = {MAKE_VARIANTS(bwData, 36)}; break;
                case 37: variants = {MAKE_VARIANTS(bwData, 37)}; break;
                case 38: variants = {MAKE_VARIANTS(bwData, 38)}; break;
                case 39: variants = {MAKE_VARIANTS(bwData, 39)}; break;
                case 40: variants = {MAKE_VARIANTS(bwData, 40)}; break;
                case 41: variants = {MAKE_VARIANTS(bwData, 41)}; break;
                case 42: variants = {MAKE_VARIANTS(bwData, 42)}; break;
                case 43: variants = {MAKE_VARIANTS(bwData, 43)}; break;
                case 44: variants = {MAKE_VARIANTS(bwData, 44)}; break;
                case 45: variants = {MAKE_VARIANTS(bwData, 45)}; break;
                case 46: variants = {MAKE_VARIANTS(bwData, 46)}; break;
                case 47: variants = {MAKE_VARIANTS(bwData, 47)}; break;
                case 48: variants = {MAKE_VARIANTS(bwData, 48)}; break;
                case 49: variants = {MAKE_VARIANTS(bwData, 49)}; break;
                case 50: variants = {MAKE_VARIANTS(bwData, 50)}; break;
                case 51: variants = {MAKE_VARIANTS(bwData, 51)}; break;
                case 52: variants = {MAKE_VARIANTS(bwData, 52)}; break;
                case 53: variants = {MAKE_VARIANTS(bwData, 53)}; break;
                case 54: variants = {MAKE_VARIANTS(bwData, 54)}; break;
                case 55: variants = {MAKE_VARIANTS(bwData, 55)}; break;
                case 56: variants = {MAKE_VARIANTS(bwData, 56)}; break;
                case 57: variants = {MAKE_VARIANTS(bwData, 57)}; break;
                case 58: variants = {MAKE_VARIANTS(bwData, 58)}; break;
                case 59: variants = {MAKE_VARIANTS(bwData, 59)}; break;
                case 60: variants = {MAKE_VARIANTS(bwData, 60)}; break;
                case 61: variants = {MAKE_VARIANTS(bwData, 61)}; break;
                case 62: variants = {MAKE_VARIANTS(bwData, 62)}; break;
                case 63: variants = {MAKE_VARIANTS(bwData, 63)}; break;
                case 64: variants = {MAKE_VARIANTS(bwData, 64)}; break;
            }
        
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

        delete inDataCol;
        delete inPosCol;
    }
    
    varex.done();
    
    return !varex.good();
}