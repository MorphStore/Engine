//
// Created by jpietrzyk on 29.05.19.
//

#include <vector/primitives/logic.h>
#include <vector/primitives/io.h>

#include <vector/scalar/extension_scalar.h>
#include <vector/scalar/primitives/logic_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/scalar/primitives/calc_scalar.h>
#include <vector/scalar/primitives/compare_scalar.h>
#include <vector/scalar/primitives/create_scalar.h>


#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/logic_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/logic_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include <vector/simd/sse/primitives/compare_sse.h>

#include <vector/datastructures/hash_based/strategies/linear_probing.h>
#include <vector/datastructures/hash_based/hash_utils.h>
#include <vector/datastructures/hash_based/hash_map.h>
#include <core/operators/general_vectorized/group_uncompr.h>

#include <vector/complex/hash.h>


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


#include <vector>
#include <algorithm>


using namespace morphstore;
using namespace vector;

// A macro expanding to an initializer list for a variant.
#define MAKE_VARIANT(ve1, ve2) \
{ \
    new varex_t::operator_wrapper \
        ::for_output_formats<uncompr_f, uncompr_f> \
        ::for_input_formats<uncompr_f>( \
            &group1< \
                    ve1, \
                    uncompr_f, \
                    uncompr_f, \
                    uncompr_f \
                   > \
    ), \
    STR_EVAL_MACROS(ve1), \
    STR_EVAL_MACROS(ve2) \
}


int main( void ) {

   using varex_t = variant_executor_helper<2, 1, size_t>::type
   ::for_variant_params<std::string, std::string>
   ::for_setting_params<size_t>;
   varex_t varex(
      {"Estimate"}, // names of the operator's additional parameters
      {"VectorExtension Process", "VectorExtension DataStructure"}, // names of the variant parameters
      {"inDataCount"} // names of the setting parameters
   );

   // Define the variants.
   const std::vector<varex_t::variant_t> variants = {
      MAKE_VARIANT(scalar<v64<uint64_t>>, sse<v128<uint64_t>>),
      MAKE_VARIANT(sse<v128<uint64_t>>, sse<v128<uint64_t>>),
#ifdef AVXTWO
      MAKE_VARIANT(scalar<v64<uint64_t>>, avx2<v256<uint64_t>>),
      MAKE_VARIANT(sse<v128<uint64_t>>, avx2<v256<uint64_t>>),
      MAKE_VARIANT(avx2<v256<uint64_t>>, avx2<v256<uint64_t>>),
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
      size_t inPosCount;
      std::tie(inDataCount, inPosCount) = sp;

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
