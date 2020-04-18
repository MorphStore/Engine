/* Benchmark application to benchmark performance impact of scalable vector lengths on certain operators.
*
*
*/
#include <iostream>

#include "benchmark_agg_sum.h"
#include "benchmark_project.h"

int main( void ) {

    using ve = tsubasa<v16384<uint64_t> >;

   
    std::cout << "Query execution started... ";
    std::cout.flush();

// files that contain benchmark results
    std::string filename1 = "agg_sum_clean";
    std::string filename2 = "project_clean";

    execute_benchmark_agg <morphstore::agg_sum_t, ve, uncompr_f>(filename1);
    execute_benchmark_project <morphstore::my_project_wit_t, ve, uncompr_f, uncompr_f, uncompr_f>(filename2);


    std::cout << "done." << std::endl << std::endl;
    


    
    return 0;
}