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
#include <vector/datastructures/hash_based/hash_set.h>
#include <core/operators/general_vectorized/group.h>

#include <vector/complex/hash.h>


#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>


#include <vector>
#include <algorithm>


#define TEST_DATA_COUNT 10000000

int main( void ) {

   using namespace vector;
   using namespace morphstore;
   std::cout << "Generating..." << std::flush;
   //column< uncompr_f > * testDataColumn = column<uncompr_f>::create_global_column(TEST_DATA_COUNT);
   const column< uncompr_f > * testDataColumnSorted = generate_sorted_unique(TEST_DATA_COUNT,1,1);


   const column<uncompr_f> * outGrCol1;
   const column<uncompr_f> * outGrColScalar1;
   const column<uncompr_f> * outExtCol1;
   const column<uncompr_f> * outExtColScalar1;
   const column<uncompr_f> * outGrCol2;
   const column<uncompr_f> * outGrColScalar2;
   const column<uncompr_f> * outExtCol2;
   const column<uncompr_f> * outExtColScalar2;
   std::cout << "Done\nVectorized..." << std::flush;

   std::tie(outGrCol1, outExtCol1) = group<avx2<v256<uint64_t>>, uncompr_f, uncompr_f>( testDataColumnSorted, TEST_DATA_COUNT );
   std::cout << "Done\nScalar..." << std::flush;
   std::tie(outGrColScalar1, outExtColScalar1) = group<processing_style_t::scalar, uncompr_f, uncompr_f>( testDataColumnSorted, TEST_DATA_COUNT );
   std::cout << "Done\nGenerating..." << std::flush;

   testDataColumnSorted = generate_with_distr(
      TEST_DATA_COUNT,
      std::uniform_int_distribution<uint64_t>(
         1,
         10
      ),
      false
   );
   std::cout << "Done\nVectorized..." << std::flush;
   std::tie(outGrCol2, outExtCol2) = group<processing_style_t::vec256, uncompr_f, uncompr_f>( testDataColumnSorted, TEST_DATA_COUNT );
   std::cout << "Done\nScalar..." << std::flush;
   std::tie(outGrColScalar2, outExtColScalar2) = group<processing_style_t::scalar, uncompr_f, uncompr_f>( testDataColumnSorted, TEST_DATA_COUNT );
   std::cout << "Done\n" << std::flush;


   const equality_check ec0(outGrCol1, outGrColScalar1);
   const equality_check ec1(outExtCol1, outExtColScalar1);
   const equality_check ec2(outGrCol2, outGrColScalar2);
   const equality_check ec3(outExtCol2, outExtColScalar2);
   const bool allGood = ec0.good() && ec1.good() && ec2.good() && ec3.good();


   return !allGood;
}
