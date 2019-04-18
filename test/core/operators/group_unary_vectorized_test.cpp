//
// Created by jpietrzyk on 18.04.19.
//

#include <core/memory/mm_glob.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/operators/vectorized/group_uncompr.h>
#include <core/utils/printing.h>

#define TEST_DATA_COUNT 1000

int main( void ) {

   using namespace morphstore;
   //column< uncompr_f > * testDataColumn = column<uncompr_f>::create_global_column(TEST_DATA_COUNT);
   const column< uncompr_f > * testDataColumnSorted = generate_sorted_unique(TEST_DATA_COUNT,1,1);


   const column<uncompr_f> * outGrCol;
   const column<uncompr_f> * outExtCol;
   std::cout << "Generated\n";
   std::tie(outGrCol, outExtCol) = group<processing_style_t::vec256, uncompr_f, uncompr_f>( testDataColumnSorted, TEST_DATA_COUNT );
   std::cout << "First done\n";
   print_columns(print_buffer_base::decimal, testDataColumnSorted, outGrCol, outExtCol, "Input", "GroupIds", "GroupEx");


   testDataColumnSorted = generate_with_distr(
      TEST_DATA_COUNT,
      std::uniform_int_distribution<uint64_t>(
         1,
         10
      ),
      false
   );
   std::cout << "Generated\n";
   std::tie(outGrCol, outExtCol) = group<processing_style_t::vec256, uncompr_f, uncompr_f>( testDataColumnSorted, TEST_DATA_COUNT );
   std::cout << "Second done\n";
   print_columns(print_buffer_base::decimal, testDataColumnSorted, outGrCol, outExtCol, "Input", "GroupIds", "GroupEx");


   return 0;
}
