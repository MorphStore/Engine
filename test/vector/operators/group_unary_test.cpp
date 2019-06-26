//
// Created by jpietrzyk on 29.05.19.
//

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>

#include <core/storage/column_gen.h>
#include <core/utils/printing.h>
#include <core/operators/scalar/group_uncompr.h>
#include <core/utils/equality_check.h>
#include "../../core/operators/operator_test_frames.h"
#include <core/utils/variant_executor.h>

#include <core/morphing/dynamic_vbp.h>

#include <vector/vector_primitives.h>
#include <vector/vector_extension_structs.h>

#include <core/operators/scalar/group_uncompr.h>
#include <core/operators/general_vectorized/group_compr.h>

#include <vector>
#include <algorithm>


using namespace morphstore;
using namespace vector;

// A macro expanding to an initializer list for a variant.
#define MAKE_VARIANT_CLASSICAL(ve1) \
{ \
    new varex_t::operator_wrapper \
        ::for_output_formats<uncompr_f, uncompr_f> \
        ::for_input_formats<uncompr_f>( \
            &group< \
                    ve1, \
                    uncompr_f, \
                    uncompr_f, \
                    uncompr_f \
                   > \
    ), \
    STR_EVAL_MACROS(ve1), \
    STR_EVAL_MACROS("uncompr_f"), \
    STR_EVAL_MACROS("uncompr_f"), \
    STR_EVAL_MACROS("uncompr_f") \
}

#define MAKE_VARIANT_VECTORIZED(ve1, form_groupid, form_groupext, form_in) \
{ \
    new varex_t::operator_wrapper \
        ::for_output_formats<form_groupid, form_groupext> \
        ::for_input_formats<form_in>( \
            &group_vec< \
                    ve1, \
                    form_groupid, \
                    form_groupext, \
                    form_in \
                   > \
    ), \
    STR_EVAL_MACROS(ve1), \
    STR_EVAL_MACROS(form_groupid), \
    STR_EVAL_MACROS(form_groupext), \
    STR_EVAL_MACROS(form_in) \
}

int main( void ) {

   using varex_t = variant_executor_helper<2, 1, size_t>::type
   ::for_variant_params<std::string, std::string, std::string, std::string>
   ::for_setting_params<size_t>;
   varex_t varex(
      {"Estimate"}, // names of the operator's additional parameters
      {"VectorExtension Process", "Format_OutGroupId", "Format_OutGroupExt", "Format_In"}, // names of the variant parameters
      {"inDataCount"} // names of the setting parameters
   );

   // Define the variants.
   const std::vector<varex_t::variant_t> variants = {
      MAKE_VARIANT_CLASSICAL(scalar<v64<uint64_t>>),

      MAKE_VARIANT_VECTORIZED(scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f),
//      MAKE_VARIANT_VECTORIZED(scalar<v64<uint64_t>>, uncompr_f, uncompr_f, SINGLE_ARG(dynamic_vbp_f<64,8,1>)),
//      MAKE_VARIANT_VECTORIZED(scalar<v64<uint64_t>>, uncompr_f, SINGLE_ARG(dynamic_vbp_f<64,8,1>), uncompr_f),
//      MAKE_VARIANT_VECTORIZED(scalar<v64<uint64_t>>, uncompr_f, SINGLE_ARG(dynamic_vbp_f<64,8,1>), SINGLE_ARG(dynamic_vbp_f<64,8,1>)),
//      MAKE_VARIANT_VECTORIZED(scalar<v64<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<64,8,1>), uncompr_f, uncompr_f),
//      MAKE_VARIANT_VECTORIZED(scalar<v64<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<64,8,1>), uncompr_f, SINGLE_ARG(dynamic_vbp_f<64,8,1>)),
//      MAKE_VARIANT_VECTORIZED(scalar<v64<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<64,8,1>), SINGLE_ARG(dynamic_vbp_f<64,8,1>), uncompr_f),
//      MAKE_VARIANT_VECTORIZED(scalar<v64<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<64,8,1>), SINGLE_ARG(dynamic_vbp_f<64,8,1>), SINGLE_ARG(dynamic_vbp_f<64,8,1>)),


//      MAKE_VARIANT_VECTORIZED(sse<v128<uint64_t>>, uncompr_f, uncompr_f, uncompr_f),
//      MAKE_VARIANT_VECTORIZED(sse<v128<uint64_t>>, uncompr_f, uncompr_f, SINGLE_ARG(dynamic_vbp_f<128,16,2>)),
//      MAKE_VARIANT_VECTORIZED(sse<v128<uint64_t>>, uncompr_f, SINGLE_ARG(dynamic_vbp_f<128,16,2>), uncompr_f),
//      MAKE_VARIANT_VECTORIZED(sse<v128<uint64_t>>, uncompr_f, SINGLE_ARG(dynamic_vbp_f<128,16,2>), SINGLE_ARG(dynamic_vbp_f<128,16,2>)),
//      MAKE_VARIANT_VECTORIZED(sse<v128<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<128,16,2>), uncompr_f, uncompr_f),
//      MAKE_VARIANT_VECTORIZED(sse<v128<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<128,16,2>), uncompr_f, SINGLE_ARG(dynamic_vbp_f<128,16,2>)),
//      MAKE_VARIANT_VECTORIZED(sse<v128<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<128,16,2>), SINGLE_ARG(dynamic_vbp_f<128,16,2>), uncompr_f),
//      MAKE_VARIANT_VECTORIZED(sse<v128<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<128,16,2>), SINGLE_ARG(dynamic_vbp_f<128,16,2>), SINGLE_ARG(dynamic_vbp_f<128,16,2>)),
#ifdef AVXTWO
//      MAKE_VARIANT_VECTORIZED(avx2<v256<uint64_t>>, uncompr_f, uncompr_f, uncompr_f),
//      MAKE_VARIANT_VECTORIZED(avx2<v256<uint64_t>>, uncompr_f, uncompr_f, SINGLE_ARG(dynamic_vbp_f<256,32,4>)),
//      MAKE_VARIANT_VECTORIZED(avx2<v256<uint64_t>>, uncompr_f, SINGLE_ARG(dynamic_vbp_f<256,32,4>), uncompr_f),
//      MAKE_VARIANT_VECTORIZED(avx2<v256<uint64_t>>, uncompr_f, SINGLE_ARG(dynamic_vbp_f<256,32,4>), SINGLE_ARG(dynamic_vbp_f<256,32,4>)),
//      MAKE_VARIANT_VECTORIZED(avx2<v256<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<256,32,4>), uncompr_f, uncompr_f),
//      MAKE_VARIANT_VECTORIZED(avx2<v256<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<256,32,4>), uncompr_f, SINGLE_ARG(dynamic_vbp_f<256,32,4>)),
//      MAKE_VARIANT_VECTORIZED(avx2<v256<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<256,32,4>), SINGLE_ARG(dynamic_vbp_f<256,32,4>), uncompr_f),
//      MAKE_VARIANT_VECTORIZED(avx2<v256<uint64_t>>, SINGLE_ARG(dynamic_vbp_f<256,32,4>), SINGLE_ARG(dynamic_vbp_f<256,32,4>), SINGLE_ARG(dynamic_vbp_f<256,32,4>)),
#endif
   };

   // Define the setting parameters.
   const std::vector<varex_t::setting_t> settingParams = {
      // inDataCount, inPosCount
      {10000},
      {501}
   };
   // Variant execution for several settings.
   for(const varex_t::setting_t sp : settingParams) {
      // Extract the individual setting parameters.
      size_t inDataCount;
//      size_t inPosCount;
      inDataCount = std::get<0>(sp);
//      std::tie(inDataCount, inPosCount) = sp;

      // Generate the data.
      varex.print_datagen_started();
      auto inDataCol = generate_with_distr(
         inDataCount,
         std::uniform_int_distribution<uint64_t>(100, 200),
         false
      );
      varex.print_datagen_done();

      // Execute the variants.
      varex.execute_variants(
         // Variants to execute
         variants,
         // Setting parameters
         inDataCount,
         // Input columns / setting
         inDataCol, 0
      );

      // Delete the generated data.
      delete inDataCol;
   }

   // Finish and print a summary.
   varex.done();

   return !varex.good();
}
