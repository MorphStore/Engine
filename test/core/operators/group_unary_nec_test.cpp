//
// Created by jpietrzyk on 18.04.19.
//

#include <core/memory/mm_glob.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/operators/general_vectorized/group_uncompr.h>
#include <core/utils/printing.h>
#include <core/utils/equality_check.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/scalar/primitives/calc_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/scalar/primitives/create_scalar.h>
#include <vector/scalar/primitives/compare_scalar.h>

#include <vector/vecprocessor/tsubasa/extension_tsubasa.h>
#include <vector/vecprocessor/tsubasa/primitives/calc_tsubasa.h>
#include <vector/vecprocessor/tsubasa/primitives/io_tsubasa.h>
#include <vector/vecprocessor/tsubasa/primitives/create_tsubasa.h>
#include <vector/vecprocessor/tsubasa/primitives/compare_tsubasa.h>
#include "operator_test_frames.h"

#define TEST_DATA_COUNT 2560000

int main( void ) {

   using namespace morphstore;
   using namespace vectorlib;
   std::cout << "Generating..." << std::flush;
   const column< uncompr_f > * testDataColumn =
      generate_repetitive_batches_with_distr(
         TEST_DATA_COUNT,
         TEST_DATA_COUNT / 100,
         std::uniform_int_distribution< uint64_t >(
            0, 100000
         ),
         false
      );

   const column<uncompr_f> * outGrCol1;
   const column<uncompr_f> * outExtCol1;
   std::cout << "Done\nScalar..." << std::flush;
   std::tie(outGrCol1, outExtCol1) =
      group1_t<
         scalar<
            v64<
               uint64_t
            >
         >,
         hash_map<
            scalar<
               v64<
                  uint64_t
               >
            >,
            multiply_mod_hash,
            size_policy_hash::EXPONENTIAL,
            scalar_key_vectorized_linear_search,
            60
         >
      >::apply(
         testDataColumn
      );

   const column<uncompr_f> * outGrCol2;
   const column<uncompr_f> * outExtCol2;
   std::cout << "Done\nVectorized..." << std::flush;
   std::tie(outGrCol2, outExtCol2) =
      group1_t<
         aurora<
            v16k<
               uint64_t
            >
         >,
         hash_map<
            aurora<
               v16k<
                  uint64_t
               >
            >,
            multiply_mod_hash,
            size_policy_hash::EXPONENTIAL,
            scalar_key_vectorized_linear_search,
            60
         >
      >::apply(
         testDataColumn
      );


   const equality_check ec0(outGrCol1, outGrCol2);
   const equality_check ec1(outExtCol1, outExtCol2);
   const bool allGood = ec0.good() && ec1.good();


   return !allGood;
}
