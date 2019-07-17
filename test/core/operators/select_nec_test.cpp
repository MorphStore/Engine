//
// Created by jpietrzyk on 15.07.19.
//

#include <core/memory/mm_glob.h>

#include "../../core/operators/operator_test_frames.h"
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>


#include <vector/vecprocessor/tsubasa/extension_tsubasa.h>
#include <vector/vecprocessor/tsubasa/primitives/calc_tsubasa.h>
#include <vector/vecprocessor/tsubasa/primitives/io_tsubasa.h>
#include <vector/vecprocessor/tsubasa/primitives/create_tsubasa.h>
#include <vector/vecprocessor/tsubasa/primitives/compare_tsubasa.h>

#include <core/operators/general_vectorized/select_uncompr.h>

#define TEST_DATA_COUNT 5120000

int main( void ) {
   using namespace morphstore;
   using namespace vectorlib;
   std::cout << "Generating..." << std::flush;

   //column< uncompr_f > * testDataColumn = column<uncompr_f>::create_global_column(TEST_DATA_COUNT);
   const column< uncompr_f > * testDataColumnSorted = generate_sorted_unique(TEST_DATA_COUNT,9,1);

   std::cout << "Done...\n";


   auto result = morphstore::select<greater, scalar<v64<uint64_t>>, uncompr_f, uncompr_f>( testDataColumnSorted, 10 );
   auto result1 = morphstore::select<greater, aurora<v16k<uint64_t>>, uncompr_f, uncompr_f>( testDataColumnSorted, 10 );

   const bool allGood =
      memcmp(result->get_data(),result1->get_data(),result1->get_count_values()*8);

   return allGood;
}