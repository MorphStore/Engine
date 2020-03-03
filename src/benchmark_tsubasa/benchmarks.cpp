#include <iostream>

#include "scalable_agg_sum.h"
#include "benchmark_project.h"

int main( void ) {

    // ************************************************************************
    // * Query execution
    // ************************************************************************
    using ve = tsubasa<v16384<uint64_t> >;

   
    std::cout << "Query execution started... ";
    std::cout.flush();

    std::string filename1 = "agg_sum";
    std::string filename2 = "project_tweak";

    // execute_benchmark_agg <morphstore::agg_sum_t, ve, uncompr_f>(filename1);
    execute_benchmark_project <morphstore::my_project_wit_t, ve, uncompr_f, uncompr_f, uncompr_f>(filename2);


    std::cout << "done." << std::endl << std::endl;
    


    
    return 0;
}