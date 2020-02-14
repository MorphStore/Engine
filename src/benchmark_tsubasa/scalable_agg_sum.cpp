#include "../../include/core/memory/mm_glob.h"
#include "../../include/core/morphing/format.h"

#include "../../include/core/operators/scalar/project_uncompr.h"
#include "../../include/core/operators/scalar/select_uncompr.h"

// #include "../../include/core/operators/general_vectorized/agg_sum_uncompr.h"
#include "../../include/core/operators/general_vectorized/agg_sum_compr.h"



#include "../../include/core/storage/column.h"
#include "../../include/core/storage/column_gen.h"
#include "../../include/core/utils/basic_types.h"
#include "../../include/core/utils/printing.h"
#include "../../include/vector/scalar/extension_scalar.h"

#include "../../include/vector/vecprocessor/tsubasa/extension_tsubasa.h"

#include <functional>
#include <iostream>
#include <random>

#include <fstream>
#include <vector>

#include <math.h>

using namespace morphstore;
using namespace vectorlib;

static inline uint64_t now();
static inline double time_elapsed_ns( uint64_t start, uint64_t end );

void execute_query(const column<uncompr_f> * const testdata);
void execute_benchmark ();

std::vector<const column<uncompr_f> * > generate_testdata();





int main( void ) {

    // ************************************************************************
    // * Query execution
    // ************************************************************************
    
   
   std::cout << "Query execution started... ";
   std::cout.flush();


   execute_benchmark();


    std::cout << "done." << std::endl << std::endl;
    
    // ************************************************************************
    // * Result output
    // ************************************************************************
   // print_columns(print_buffer_base::decimal, generate_testdata().front(), "Testdata");


    
    return 0;
}


static inline uint64_t now() {
   uint64_t  ret;
   asm volatile( "smir %0, %%usrcc":"=r"(ret) );
   return ret;
}
static inline double time_elapsed_ns( uint64_t start, uint64_t end ) {
   return ( (double)( end - start ) ) / 1.4;
}

void execute_query(const column<uncompr_f> * const testdata){
   using ve = tsubasa<v16384<uint64_t> >;

   std::ofstream resultfile;
   resultfile.open("/home/mundt/Results/results.csv", std::ofstream::out | std::ofstream::app);
   resultfile << "Size in MiB" << "," << "Duration in ms" << "," << "Size of column in kiB" <<"\n"; 

   uint64_t result = 0;

   const size_t inner = 1;

   const column <uncompr_f> * results [inner+1];

   const column <uncompr_f> * reference = morphstore::agg_sum<scalar<v64<uint64_t>>, uncompr_f>(testdata);

   //Warm-Up
   results[0] = morphstore::agg_sum<ve, uncompr_f>(testdata);


   for (int i =0; i < 5; i++){
   
      uint64_t start = now();
      for(size_t j=0; j<inner; j++){
         // results[j+1] = morphstore::agg_sum<scalar<v64<uint64_t>>, uncompr_f>(testdata);
         results[j+1] = morphstore::agg_sum<ve, uncompr_f>(testdata);

      }

      double duration = time_elapsed_ns(start, now())/inner;


      for(size_t j=0; j<inner; j++){
         const uint8_t  * column_ptr = results[j+1]->get_data();
         const uint64_t * result_ptr = reinterpret_cast<const uint64_t  *> (column_ptr);

         std::cout << "Ergebnis: " << (*result_ptr) << std::endl;
         result |= (*result_ptr);
         
         delete results[j+1];
      }
      resultfile << ((testdata->get_count_values()  *8) >> 20) << "," << duration/1000000 << "," << (testdata->get_size_used_byte() >> 10)  << "\n"; 
   } 

   delete results[0];

   
   resultfile.close();
   const uint8_t  * ref_ptr = reference->get_data();
   const uint64_t * result_ref_ptr = reinterpret_cast<const uint64_t  *> (ref_ptr);

   std::cout << "Result: " << result << std::endl;
   std::cout << "Result-ref: " << (*result_ref_ptr) << std::endl;

   delete reference;


}

void execute_benchmark(){
   // std::vector<size_t> testDataSizes {1_MB, 2_MB, 4_MB, 8_MB, 16_MB, 32_MB, 64_MB, 128_MB, 256_MB, 512_MB, 1_GB, 2_GB};
   std::vector<size_t> testDataSizes {4_KB};

   // std::vector<size_t> testDataSizes {16_KB, 32_KB, 64_KB, 128_KB, 256_KB, 512_KB};

   for(const size_t& testsize: testDataSizes){

      // ************************************************************************
      // * Generation of the synthetic base data
      // ************************************************************************
      
      std::cout << "Base data generation started... ";
      std::cout.flush();
         

      std::cout << "done." << std::endl;
      std::cout.flush();
      // print_columns(print_buffer_base::decimal, baseCol1, "Test column");
      std::cout.flush();


      size_t countValues = (testsize >> 3) + 10; // count = size in Byte divided by sizeof base-type

      const column<uncompr_f> *  baseCol = generate_with_distr(
      countValues,
      std::uniform_int_distribution<uint64_t>(1, 50),
      true
      );

      execute_query(baseCol);
      delete baseCol;

   }
}
