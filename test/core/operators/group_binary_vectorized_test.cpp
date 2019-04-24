//
// Created by jpietrzyk on 18.04.19.
//
#include <iostream>
#include <core/memory/mm_glob.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/operators/vectorized/group_uncompr.h>
#include <core/operators/scalar/group_uncompr.h>
#include <core/utils/printing.h>
#include <core/utils/equality_check.h>
#include "operator_test_frames.h"
#define TEST_DATA_COUNT 10000000

int main( void ) {

   using namespace morphstore;
   //column< uncompr_f > * testDataColumn = column<uncompr_f>::create_global_column(TEST_DATA_COUNT);
   const column< uncompr_f > * testDataColumnSorted1 = generate_with_distr(
      TEST_DATA_COUNT,
      std::uniform_int_distribution<uint64_t>(
         1,
         10
      ),
      false
   );
   const column< uncompr_f > * testDataColumnSorted2 = generate_with_distr(
      TEST_DATA_COUNT,
      std::uniform_int_distribution<uint64_t>(
         1,
         20
      ),
      false
   );


   const column<uncompr_f> * outGrCol;
   const column<uncompr_f> * outGrColTmp;
   const column<uncompr_f> * outGrColScalar;
   const column<uncompr_f> * outGrColScalarTmp;
   const column<uncompr_f> * outExtCol;
   const column<uncompr_f> * outExtColScalar;

   std::tie(outGrColScalarTmp, outExtColScalar) = group<processing_style_t::scalar, uncompr_f, uncompr_f>( testDataColumnSorted1, TEST_DATA_COUNT );
//   std::cout << "Scalar Result:\n";
//   print_columns(print_buffer_base::decimal, testDataColumnSorted1, outGrColScalar, outExtColScalar, "Input", "GroupIds", "GroupEx");
   std::tie(outGrColTmp, outExtCol) = group<processing_style_t::vec256, uncompr_f, uncompr_f>( testDataColumnSorted1, TEST_DATA_COUNT );
//   std::cout << "Vec Result:\n";
//   print_columns(print_buffer_base::decimal, testDataColumnSorted1, outGrCol, outExtCol, "Input", "GroupIds", "GroupEx");

   std::tie(outGrColScalar, outExtColScalar) = group<processing_style_t::scalar, uncompr_f, uncompr_f>( outGrColScalarTmp, testDataColumnSorted2, TEST_DATA_COUNT );
//   std::cout << "Scalar Result:\n";
//   print_columns(print_buffer_base::decimal, testDataColumnSorted2, outGrColScalarTmp, outGrColScalar, outExtColScalar, "Input Data", "Input GroupIds", "Result GroupIds", "Result GroupEx");
   std::tie(outGrCol, outExtCol) = group<processing_style_t::vec256, uncompr_f, uncompr_f>( outGrColTmp, testDataColumnSorted2, TEST_DATA_COUNT );
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
