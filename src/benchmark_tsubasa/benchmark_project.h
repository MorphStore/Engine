/*
*  Used to benchmark the performance impact of scalable vector lengths on the project operator.
*
*/


#ifndef BENCHMARK_PROJECT_H
#define BENCHMARK_PROJECT_H

#include "../../include/core/memory/mm_glob.h"
#include "../../include/core/morphing/format.h"

// #include "../../include/core/operators/scalar/project_uncompr.h"
#include "../../include/core/operators/scalar/select_uncompr.h"

// #include "../../include/core/operators/general_vectorized/agg_sum_uncompr.h"
#include "../../include/core/operators/general_vectorized/agg_sum_compr.h"
#include "../../include/core/operators/general_vectorized/project_compr.h"




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
void execute_benchmark_project (std::string filename);

std::vector<const column<uncompr_f> * > generate_testdata();


template <template <class, class ...> class op,  typename ve, class ... op_args >
void execute_query(const column<uncompr_f> * const baseCol, const column<uncompr_f> * const posCol, std::string filename){

   std::ofstream resultfile;
   resultfile.open("/home/mundt/Results/"+filename+".csv", std::ofstream::out | std::ofstream::app);
   // resultfile << "Size_in_MB" << "," << "Duration_in_ms" << "," << "Size_in_kB" << "," << "Scalable" "\n"; 

   uint64_t result = 0;
// Number of repetitions of the outer loop.
   size_t inner;
   const size_t inner_max = 1000;

   if(posCol->get_size_used_byte() < 4_MB){
      inner = 1000;
   } else if(posCol->get_size_used_byte() < 32_MB){
      inner = 100;
   } else {
      inner = 10;
   }
// Number of repetitions of the outer loop.
   const size_t outer = 20;


   const column <uncompr_f> * results [inner_max+1];

   const column <uncompr_f> * reference = op<scalar<v64<uint64_t>>, op_args ...>::apply(baseCol, posCol);

   //Warm-Up
   results[0] = op<ve, op_args ...>::apply(baseCol, posCol);




   for (size_t i =0; i < outer; i++){
   

      std::cout << "Start Benchmarking" << std::endl;
      std::cout << "Inner: " << inner << std::endl;

      // auto begin = std::chrono::system_clock::now();
      uint64_t start = now();
      // Inner loop. Operator call is repeated "inner" times.
      // "duration" is the time it takes to complete all these calls.
      for(size_t j=0; j<inner; j++){
         // results[j+1] = morphstore::agg_sum<scalar<v64<uint64_t>>, uncompr_f>(testdata);
         results[j+1] = op<ve, op_args ...>::apply(baseCol, posCol);

      }

      double duration = time_elapsed_ns(start, now());



      for(size_t j=0; j<inner; j++){
         const uint8_t  * column_ptr = results[j+1]->get_data();
         const uint64_t * result_ptr = reinterpret_cast<const uint64_t  *> (column_ptr);

         // std::cout << "Ergebnis: " << (*result_ptr) << std::endl;
         result |= (*result_ptr);
         
         delete results[j+1];
      }
      resultfile << posCol->get_size_used_byte() << ";" << duration << ";" <<  inner  << ";"  << "yes" << "\n"; 
   } 


   
   resultfile.close();
   const uint8_t  * ref_ptr = reference->get_data();
   const uint64_t * result_ref_ptr = reinterpret_cast<const uint64_t  *> (ref_ptr);

   std::cout << "Result: " << result << std::endl;
   std::cout << "Result-ref: " << (*result_ref_ptr) << std::endl;
   // print_columns(print_buffer_base::decimal, reference, "Reference Column");
   // print_columns(print_buffer_base::decimal, results[0], "Vectorized Column");

   delete results[0];


   delete reference;


}






template <template <class, class ...> class op, typename ve, class ... op_args >
void execute_benchmark_project (std::string filename  ){
   // std::vector<size_t> testDataSizes { 16_MB, 32_MB, 64_MB, 128_MB, 256_MB, 512_MB, 1_GB, 2_GB};
   // std::vector<size_t> testDataSizes {1_MB, 2_MB, 4_MB, 8_MB};
   // std::vector<size_t> testDataSizes { 16_MB, 32_MB, 64_MB, 128_MB, 256_MB, 512_MB};

   std::vector<size_t> testDataSizes {16_KB, 32_KB, 64_KB, 128_KB, 256_KB, 512_KB, 1_MB, 2_MB, 4_MB, 8_MB, 
                                       16_MB, 32_MB, 64_MB, 128_MB, 256_MB, 512_MB};

   // std::vector<size_t> testDataSizes { 2_KB};


   std::cout << "Base data generation started... ";
   std::cout.flush();
      


   size_t baseColSize = 2_GB;
   size_t baseColCount = baseColSize >> 3;

    const column<uncompr_f> *  baseCol = generate_with_distr(
    (baseColCount),
    std::uniform_int_distribution<uint64_t>(1, 50),
    true
    );
   std::cout << "done." << std::endl;

   for(const size_t& testsize: testDataSizes){

      // ************************************************************************
      // * Generation of the synthetic base data
      // ************************************************************************
      
      std::cout << "PosCol data generation started... ";
      std::cout.flush();


      size_t countValues = (testsize >> 3) + 255; // count = size in Byte divided by sizeof base-type (uint64_t)

      // const column<uncompr_f> *  posCol = generate_with_distr(
      // countValues,
      // std::uniform_int_distribution<uint64_t>(1, 50),
      // true
      // );

      const column<uncompr_f> *  posCol = generate_sorted_unique_extraction(countValues, baseColCount);


      std::cout << "done." << std::endl;
      std::cout.flush();
      // print_columns(print_buffer_base::decimal, baseCol1, "Test column");
      std::cout.flush();

      execute_query<op, ve, op_args ...>(baseCol, posCol, filename);
      delete posCol;

   }
}

// int main (void) {

//     using ve = tsubasa<v16384<uint64_t>>;

//     std::string filename = "project";
    
//     execute_benchmark<morphstore::my_project_wit_t, ve, uncompr_f, uncompr_f, uncompr_f>(filename);



//     return 0;
// }

#endif //BENCHMARK_PROJECT_H