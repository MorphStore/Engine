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

#include <core/operators/general_vectorized/calc_uncompr.h>

#define TEST_DATA_COUNT 100

int main( void ) {
   using namespace morphstore;
   using namespace vector;
   std::cout << "Generating..." << std::flush;
   //column< uncompr_f > * testDataColumn = column<uncompr_f>::create_global_column(TEST_DATA_COUNT);
   const column< uncompr_f > * testDataColumnSorted = generate_sorted_unique(TEST_DATA_COUNT,1,1);
   const column< uncompr_f > * testDataColumnSorted2 = generate_sorted_unique(TEST_DATA_COUNT,1,1);
   std::cout << "Done...\n";


   auto result = compare_binary<avx2<v256<uint64_t>>, 64,equal>::apply( testDataColumnSorted, testDataColumnSorted2 );
   auto result1 = compare_binary<sse<v128<uint64_t>>, 64,equal>::apply( testDataColumnSorted, testDataColumnSorted2 );

   const bool allGood =
      memcmp(result->get_data(),result1->get_data(),(int)(TEST_DATA_COUNT/8));

   std::cout << "result size in Byte: " <<  TEST_DATA_COUNT/8 << ", result[0]: " << ((uint32_t*)result->get_data())[0] << ", result1[0]: " << ((uint32_t*)result1->get_data())[0] << std::endl; 
   std::cout << "result[1]: " << ((uint32_t*)result->get_data())[1] << ", result1[1]: " << ((uint32_t*)result1->get_data())[1] << std::endl; 
   std::cout << "result[2]: " << ((uint32_t*)result->get_data())[2] << ", result1[2]: " << ((uint32_t*)result1->get_data())[2] << std::endl; 
   return allGood;
}