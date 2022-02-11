#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>

#include <vector/vector_extension_structs.h>
#include <vector/gpu/primitives/calc_gpu.h>
#include <vector/gpu/primitives/compare_gpu.h>
#include <vector/gpu/primitives/manipulate_gpu.h>
#include <vector/gpu/primitives/create_gpu.h>
#include <vector/gpu/primitives/extract_gpu.h>
#include <vector/gpu/primitives/logic_gpu.h>
#include <vector/gpu/primitives/io_gpu.h>

// #include "core/operators/reference/agg_sum_all.h"
#include "core/operators/uncompr/agg_sum_all.h"
// #include <core/operators/general_vectorized/agg_sum_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace vectorlib;
using namespace morphstore;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(void)
{
   std::cout << "Base data generation started... ";
   std::cout.flush();

   const size_t array_size = 32;
   // const uint64_t array_bytes = array_size * sizeof(uint64_t);
   const column<uncompr_f> * const baseCol1 = generate_with_distr(
      array_size,
      std::uniform_int_distribution<uint64_t>(0, 5),
      true,
      2
   );
   print_columns<uint64_t>(print_buffer_base::decimal, baseCol1, "baseCol1");
   using ps1 = scalar<v64<uint64_t> >; //gpu<v2048<uint64_t>>;
   IMPORT_VECTOR_BOILER_PLATE(ps1)
   std::cout << "Query execution started... ";
   std::cout.flush();
   auto i1 = agg_sum_all<ps1, uncompr_f>(baseCol1);
   std::cout << "done." << std::endl << std::endl;
   print_columns(print_buffer_base::decimal, i1, "SUM(baseCol1)");

   return 0;
}

// int main(void)
// {
//    using ps1 = gpu<v2048<uint64_t>>;
//    const uint64_t array_size = 32;
//    const uint64_t array_bytes = array_size * sizeof(uint64_t);
//    uint64_t *h_arr1;
//    uint64_t *d_arr1;
//     uint64_t h_result = 0;
//     uint64_t *d_result;
//     gpuErrchk( cudaMalloc((void**)&d_result, sizeof(uint64_t)) );

//    gpuErrchk( cudaMallocHost((void**)&h_arr1, array_bytes) );
//    gpuErrchk( cudaMalloc((void**)&d_arr1, array_bytes) );
//    float ms;
//     cudaEvent_t startEvent, stopEvent;
//     gpuErrchk( cudaEventCreate(&startEvent) );
//     gpuErrchk( cudaEventCreate(&stopEvent) );
//    srand(time(NULL));
//    for(uint64_t i = 0; i < array_size; i++){
//       h_arr1[i] = rand() % 10;//range 0 to 9
//    }
//    std::cout<<"h_arr1:"<<std::endl;
//    for(uint64_t i = 0; i < array_size; i++){
//       std::cout<<h_arr1[i]<< " ";
//    }
//    std::cout << std::endl;
//     gpuErrchk( cudaMemcpyAsync(d_arr1, h_arr1, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
//    gpuErrchk( cudaEventRecord(startEvent,0) );
//    d_result = hadd<ps1, ps1::vector_helper_t::granularity::value>::apply(d_arr1);
//    gpuErrchk( cudaEventRecord(stopEvent, 0) );
//     gpuErrchk( cudaEventSynchronize(stopEvent) );
//     gpuErrchk( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
//    printf("Time for kernel launch (ms): %f\n", ms);
//    gpuErrchk( cudaMemcpyAsync(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost) );
//    gpuErrchk( cudaDeviceSynchronize() );
//    std::cout<<"Output: "<< h_result << std::endl;
//    return 0;
// }
