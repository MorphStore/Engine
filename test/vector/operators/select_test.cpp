/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include "../../core/operators/operator_test_frames.h"
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>

#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>



#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include <vector/simd/sse/primitives/compare_sse.h>

#include <core/operators/general_vectorized/select_uncompr.h>

#define TEST_DATA_COUNT 100

int main( void ) {
   using namespace morphstore;
   using namespace vectorlib;
   std::cout << "Generating..." << std::flush;
   //column< uncompr_f > * testDataColumn = column<uncompr_f>::create_global_column(TEST_DATA_COUNT);
   const column< uncompr_f > * testDataColumnSorted = generate_sorted_unique(TEST_DATA_COUNT,9,1);
   
   std::cout << "Done...\n";


   auto result = morphstore::select<greater, avx2<v256<uint64_t>>, uncompr_f, uncompr_f>( testDataColumnSorted, 10 );
   auto result1 = morphstore::select<greater, sse<v128<uint64_t>>, uncompr_f, uncompr_f>( testDataColumnSorted, 10 );

   const bool allGood =
      memcmp(result->get_data(),result1->get_data(),result1->get_count_values()*8);

   
   return allGood;
}