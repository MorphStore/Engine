
#include <iostream>

#include "benchmark_util.h"







int main( void ) {

    // ************************************************************************
    // * Query execution
    // ************************************************************************
       using ve = tsubasa<v16384<uint64_t> >;

   
   std::cout << "Query execution started... ";
   std::cout.flush();

   std::string filename = "results_agg_sum";

   execute_benchmark<morphstore::agg_sum_t, ve, uncompr_f>(filename);


    std::cout << "done." << std::endl << std::endl;
    


    
    return 0;
}





