//
// Created by jpietrzyk on 05.06.19.
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
#include <vector/datastructures/hash_based/hash_binary_key_map.h>
#include <core/operators/general_vectorized/group.h>

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


#include <vector>
#include <algorithm>

#define TEST_DATA_COUNT 10000000

int main( void ) {

   using namespace morphstore;
   using namespace vector;
   //column< uncompr_f > * testDataColumn = column<uncompr_f>::create_global_column(TEST_DATA_COUNT);
   const column <uncompr_f> *testDataColumnSorted1 = generate_with_distr(
      TEST_DATA_COUNT,
      std::uniform_int_distribution<uint64_t>(
         1,
         10
      ),
      false
   );
   const column <uncompr_f> *testDataColumnSorted2 = generate_with_distr(
      TEST_DATA_COUNT,
      std::uniform_int_distribution<uint64_t>(
         1,
         20
      ),
      false
   );


   const column <uncompr_f> *outGrCol;
   const column <uncompr_f> *outGrColTmp;
   const column <uncompr_f> *outGrColScalar;
   const column <uncompr_f> *outGrColScalarTmp;
   const column <uncompr_f> *outExtCol;
   const column <uncompr_f> *outExtColScalar;

   std::tie(outGrColScalarTmp, outExtColScalar) = group < scalar < v64 < uint64_t >>, uncompr_f, uncompr_f >
                                                                                                 (testDataColumnSorted1, TEST_DATA_COUNT);
//   std::cout << "Scalar Result:\n";
//   print_columns(print_buffer_base::decimal, testDataColumnSorted1, outGrColScalar, outExtColScalar, "Input", "GroupIds", "GroupEx");
   std::tie(outGrColTmp, outExtCol) =
      group1<
         uncompr_f,
         sse<v128<uint64_t>>,
         hash_map<
            avx2<v256<uint64_t>>,
            multiply_mod_hash,
            size_policy_hash::EXPONENTIAL,
            scalar_key_vectorized_linear_search,
            60
         >
      >(testDataColumnSorted1, TEST_DATA_COUNT);


//   std::cout << "Vec Result:\n";
//   print_columns(print_buffer_base::decimal, testDataColumnSorted1, outGrCol, outExtCol, "Input", "GroupIds", "GroupEx");

   std::tie(outGrColScalar, outExtColScalar) = group < scalar < v64 < uint64_t >>, uncompr_f, uncompr_f >
                                                                                              (outGrColScalarTmp, testDataColumnSorted2, TEST_DATA_COUNT);
//   std::cout << "Scalar Result:\n";
//   print_columns(print_buffer_base::decimal, testDataColumnSorted2, outGrColScalarTmp, outGrColScalar, outExtColScalar, "Input Data", "Input GroupIds", "Result GroupIds", "Result GroupEx");
   std::tie(outGrCol, outExtCol) =
      group1<
         uncompr_f,
         avx2<v256<uint64_t>>,
         hash_binary_key_map<
            avx2<v256<uint64_t>>,
            multiply_mod_hash,
            size_policy_hash::EXPONENTIAL,
            scalar_key_vectorized_linear_search,
            60
         >
      >(outGrColTmp, testDataColumnSorted2, TEST_DATA_COUNT);

//   std::cout << "Vec Result:\n";
//   print_columns(print_buffer_base::decimal, testDataColumnSorted2, outGrColTmp, outGrCol, outExtCol, "Input Data", "Input GroupIds", "Result GroupIds", "Result GroupEx");

   const equality_check ec0(outGrColScalar, outGrCol);
   const equality_check ec1(outExtColScalar, outExtCol);
   const bool allGood = ec0.good() && ec1.good();


   print_check("GroupId", ec0);
   print_check("GroupExt", ec1);
   print_overall(allGood);
   return !allGood;
}