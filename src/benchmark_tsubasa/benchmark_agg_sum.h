#ifndef SCALABLE_AGG_SUM_H
#define SCALABLE_AGG_SUM_H

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

#include <chrono>

#include "time_tsubasa.h"

template <template <class, class ...> class op, typename ve, class ... op_args >
void execute_query(const column<uncompr_f> * const testdata, std::string filename);

template <template <class, class ...> class op, typename ve, class ... op_args >
void execute_benchmark_agg (std::string filename);

std::vector<const column<uncompr_f> * > generate_testdata();


template <template <class, class ...> class op,  typename ve, class ... op_args >
void execute_query(const column<uncompr_f> * const testdata, std::string filename){

   std::ofstream resultfile;
   resultfile.open("/home/mundt/Results/"+filename+".csv", std::ofstream::out | std::ofstream::app);
   // resultfile << "Size_in_MiB" << "," << "Duration_in_ms" << "," << "Size_in_kiB" << "," << "Scalable" "\n"; 

   uint64_t result = 0;

   size_t inner;
   const size_t inner_max = 1000;

   if(testdata->get_size_used_byte() < 4_MB){
      inner = 1000;
   } else if(testdata->get_size_used_byte() < 32_MB){
      inner = 100;
   } else {
      inner = 10;
   }

   const size_t outer = 20;


   const column <uncompr_f> * results [inner_max+1];

   const column <uncompr_f> * reference = op<scalar<v64<uint64_t>>, op_args ...>::apply(testdata);

   //Warm-Up
   results[0] = op<ve, op_args ...>::apply(testdata);


   for (size_t i =0; i < outer; i++){
   

      std::cout << "Start Benchmarking" << std::endl;
      // auto begin = std::chrono::system_clock::now();
      uint64_t start = now();
      for(size_t j=0; j<inner; j++){
         // results[j+1] = morphstore::agg_sum<scalar<v64<uint64_t>>, uncompr_f>(testdata);
         results[j+1] = op<ve, uncompr_f>::apply(testdata);

      }

      double duration = time_elapsed_ns(start, now());



      for(size_t j=0; j<inner; j++){
         const uint8_t  * column_ptr = results[j+1]->get_data();
         const uint64_t * result_ptr = reinterpret_cast<const uint64_t  *> (column_ptr);

         // std::cout << "Ergebnis: " << (*result_ptr) << std::endl;
         result |= (*result_ptr);
         
         delete results[j+1];
      }
      resultfile << testdata->get_size_used_byte() << ";" << duration << ";" <<  inner  << ";" <<  result << ";" << "no" << "\n"; 
   } 

   delete results[0];

   
   resultfile.close();
   const uint8_t  * ref_ptr = reference->get_data();
   const uint64_t * result_ref_ptr = reinterpret_cast<const uint64_t  *> (ref_ptr);

   std::cout << "Result: " << result << std::endl;
   std::cout << "Result-ref: " << (*result_ref_ptr) << std::endl;

   delete reference;


}






template <template <class, class ...> class op, typename ve, class ... op_args >
void execute_benchmark_agg(std::string filename  ){
   // std::vector<size_t> testDataSizes { 16_MB, 32_MB, 64_MB, 128_MB, 256_MB, 512_MB, 1_GB, 2_GB};
   // std::vector<size_t> testDataSizes {1_MB, 2_MB, 4_MB, 8_MB};

   std::vector<size_t> testDataSizes {16_KB, 32_KB, 64_KB, 128_KB, 256_KB, 512_KB, 1_MB, 2_MB, 4_MB, 8_MB,
                                       16_MB, 32_MB, 64_MB, 128_MB, 256_MB, 512_MB, 1_GB, 2_GB};
   // std::vector<size_t> testDataSizes {8_KB};


   for(const size_t& testsize: testDataSizes){

      // ************************************************************************
      // * Generation of the synthetic base data
      // ************************************************************************
      
      std::cout << "Base data generation started... ";
      std::cout.flush();
         

      std::cout << "done." << std::endl;
      std::cout.flush();
      std::cout.flush();


      size_t countValues = (testsize >> 3) + 255; // count = size in Byte divided by sizeof base-type

      const column<uncompr_f> *  baseCol = generate_with_distr(
      countValues,
      std::uniform_int_distribution<uint64_t>(1, 500),
      true
      );

    //   print_columns(print_buffer_base::decimal, baseCol, "Test column");


      execute_query<op, ve, op_args ...>(baseCol, filename);
      delete baseCol;

   }
}

// int main( void ) {

//     // ************************************************************************
//     // * Query execution
//     // ************************************************************************
//        using ve = tsubasa<v16384<uint64_t> >;

   
//    std::cout << "Query execution started... ";
//    std::cout.flush();

//    std::string filename = "agg_sum";

//    execute_benchmark<morphstore::agg_sum_t, ve, uncompr_f>(filename);


//     std::cout << "done." << std::endl << std::endl;
    


    
//     return 0;
// }

#endif //SCALABLE_AGG_SUM_H





