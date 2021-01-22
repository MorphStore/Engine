#include <core/memory/mm_glob.h>
#include <vector/vector_extension_structs.h>
//#include <vector/vector_primitives.h>
#include <vector/gpu/primitives/calc_gpu.h>
#include <vector/gpu/primitives/compare_gpu.h>
#include <vector/gpu/primitives/manipulate_gpu.h>
#include <vector/gpu/primitives/create_gpu.h>
#include <vector/gpu/primitives/extract_gpu.h>
#include <vector/gpu/primitives/logic_gpu.h>
#include <vector/gpu/primitives/io_gpu.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace vectorlib;
//using namespace morphstore;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//testing compare
int main(void)
{
    using ps1 = gpu<v2048<uint64_t>>;
    const uint64_t array_size = 32;
    const uint64_t array_bytes = array_size * sizeof(uint64_t);
    uint64_t h_arr1[array_size], h_arr2[array_size];
    uint32_t h_result = 0;
    uint64_t *d_arr1, *d_arr2;
    uint32_t *d_result;
    gpuErrchk( cudaMalloc((void**)&d_arr1, array_bytes) );
    gpuErrchk( cudaMalloc((void**)&d_arr2, array_bytes) );
    gpuErrchk( cudaMalloc((void**)&d_result, sizeof(uint32_t)) );

    float ms;
    cudaEvent_t startEvent, stopEvent;
    gpuErrchk( cudaEventCreate(&startEvent) );
    gpuErrchk( cudaEventCreate(&stopEvent) );
    //generate input array
    srand(time(NULL));
    for(uint64_t i = 0; i < array_size; i++){
        h_arr1[i] = rand() % 10;//range 0 to 9
        h_arr2[i] = rand() % 10;//range 0 to 9
    }
    std::cout<<"h_arr1:"<<std::endl;
    for(uint64_t i = 0; i < array_size; i++){
        std::cout<<h_arr1[i]<< " ";
    }
    std::cout << std::endl;
    std::cout<<"h_arr2:"<<std::endl;
    for(uint64_t i = 0; i < array_size; i++){
        std::cout<<h_arr2[i]<< " ";
    }
    std::cout << std::endl;
    gpuErrchk( cudaMemcpy(d_arr1, h_arr1, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
    gpuErrchk( cudaMemcpy(d_arr2, h_arr2, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
    gpuErrchk( cudaEventRecord(startEvent,0) );
    d_result = greaterequal<ps1, ps1::vector_helper_t::granularity::value>::apply(d_arr1, d_arr2);
    gpuErrchk( cudaEventRecord(stopEvent, 0) );
    gpuErrchk( cudaEventSynchronize(stopEvent) );
    gpuErrchk( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("Time for kernel launch (ms): %f\n", ms);
    gpuErrchk( cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost) ); //transfer the ouput array to the CPU
    gpuErrchk( cudaDeviceSynchronize() );
    std::cout<<"Output array:"<< h_result <<std::endl;
    std::cout << std::endl;
    return 0;
}

//testing count_matches, count_leading_zero
// int main(void)
// {
//     using ps1 = gpu<v2048<uint64_t>>;
//     uint32_t h_mask = 57;
//     uint8_t h_result = 0;
//     uint32_t *d_mask;
//     uint8_t* d_result;
//     gpuErrchk( cudaMalloc((void**)&d_mask, sizeof(uint32_t)) );
//     gpuErrchk( cudaMalloc((void**)&d_result, sizeof(uint8_t)) );
//     float ms;
//     cudaEvent_t startEvent, stopEvent;
//     gpuErrchk( cudaEventCreate(&startEvent) );
//     gpuErrchk( cudaEventCreate(&stopEvent) );
//     gpuErrchk( cudaMemcpy(d_mask, &h_mask, sizeof(uint32_t), cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
//     gpuErrchk( cudaEventRecord(startEvent,0) );
//     d_result = count_matches<ps1>::apply(d_mask);
//     // d_result = count_leading_zero<ps1>::apply(d_mask);
//     gpuErrchk( cudaEventRecord(stopEvent, 0) );
//     gpuErrchk( cudaEventSynchronize(stopEvent) );
//     gpuErrchk( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
//     printf("Time for kernel launch (ms): %f\n", ms);
//     gpuErrchk( cudaMemcpy(&h_result, d_result, sizeof(uint8_t), cudaMemcpyDeviceToHost) ); //transfer the ouput array to the CPU
//     gpuErrchk( cudaDeviceSynchronize() );
//     std::cout<<"Output:"<< unsigned(h_result) <<std::endl;
//     return 0;
// }

//test bitwise_and for mask
// int main(void)
// {
//     using ps1 = gpu<v2048<uint64_t>>;
//     const uint32_t h_mask1 = 30;
//     const uint32_t h_mask2 = 7;
//     uint32_t h_result = 0;
//     uint32_t *d_mask1, *d_mask2, *d_result;
//     gpuErrchk( cudaMalloc((void**)&d_mask1, sizeof(uint32_t)) );
//     gpuErrchk( cudaMalloc((void**)&d_mask2, sizeof(uint32_t)) );
//     gpuErrchk( cudaMalloc((void**)&d_result, sizeof(uint32_t)) );
//     float ms;
//     cudaEvent_t startEvent, stopEvent;
//     gpuErrchk( cudaEventCreate(&startEvent) );
//     gpuErrchk( cudaEventCreate(&stopEvent) );
//     gpuErrchk( cudaMemcpy(d_mask1, &h_mask1, sizeof(uint32_t), cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
//     gpuErrchk( cudaMemcpy(d_mask2, &h_mask2, sizeof(uint32_t), cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
//     gpuErrchk( cudaEventRecord(startEvent,0) );
//     d_result = bitwise_and<ps1,ps1::vector_helper_t::granularity::value>(d_mask1, d_mask2);
//     gpuErrchk( cudaEventRecord(stopEvent, 0) );
//     gpuErrchk( cudaEventSynchronize(stopEvent) );
//     gpuErrchk( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
//     printf("Time for kernel launch (ms): %f\n", ms);
//     gpuErrchk( cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost) ); //transfer the ouput array to the CPU
//     gpuErrchk( cudaDeviceSynchronize() );
//     std::cout<<"Output:"<< h_result <<std::endl;
// }

//testing add
// int main(void)
// {
//     //for(int num_blocks = 1; num_blocks <= 524288; num_blocks<<=1){
//         using ps1 = gpu<v2048<uint64_t>>;
//     	//const uint64_t array_size = 32;
//         //16,777,216 values
//     	const uint64_t elements_processed_per_launch = 32;//*num_blocks;
//         const uint64_t number_of_kernel_launches = 1;//16777216 / elements_processed_per_launch;
//     	const uint64_t array_size = elements_processed_per_launch * number_of_kernel_launches;

//     	const uint64_t array_bytes = array_size * sizeof(uint64_t);
//     	uint64_t *h_arr1, *h_arr2;
//     	uint64_t *d_arr1, *d_arr2, *d_result;
//     	gpuErrchk( cudaMallocHost((void**)&h_arr1, array_bytes) );
//     	gpuErrchk( cudaMallocHost((void**)&h_arr2, array_bytes) );
//     	gpuErrchk( cudaMalloc((void**)&d_arr1, array_bytes) );
//     	gpuErrchk( cudaMalloc((void**)&d_arr2, array_bytes) );
//         gpuErrchk( cudaMalloc((void**)&d_result, array_bytes) );

//         float ms;
//         cudaEvent_t startEvent, stopEvent;
//         gpuErrchk( cudaEventCreate(&startEvent) );
//         gpuErrchk( cudaEventCreate(&stopEvent) );

//         //generate input array
//     	srand(time(NULL));
//     	for(uint64_t i = 0; i < array_size; i++){
//     		//can probably be generated faster with curand
//     		h_arr1[i] = rand() % 10;//range 0 to 9
//     		h_arr2[i] = rand() % 10;//range 0 to 9
//     	}

//     	std::cout<<"h_arr1:"<<std::endl;
//     	for(uint64_t i = 0; i < array_size; i++){
//     		std::cout<<h_arr1[i]<< " ";
//     	}
//     	std::cout << std::endl;

//     	std::cout<<"h_arr2:"<<std::endl;
//     	for(uint64_t i = 0; i < array_size; i++){
//     		std::cout<<h_arr2[i]<< " ";
//     	}
//     	std::cout << std::endl;

//         gpuErrchk( cudaMemcpyAsync(d_arr1, h_arr1, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
//         gpuErrchk( cudaMemcpyAsync(d_arr2, h_arr2, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU

//         gpuErrchk( cudaEventRecord(startEvent,0) );
//         for(int i = 0; i < number_of_kernel_launches; i++){
//         	int offset = elements_processed_per_launch * i;
//     		//vectorlib::add<ps1, ps1::vector_helper_t::granularity::value>::apply(d_arr1 + offset, d_arr2 + offset, num_blocks); //calls add, inside add is the kernel launch
//             //d_result = vectorlib::add<ps1, ps1::vector_helper_t::granularity::value>::apply(d_arr1 + offset, d_arr2 + offset);
//             d_result = bitwise_or<ps1, ps1::vector_helper_t::granularity::value>(d_arr1 + offset, d_arr2 + offset);
//         }
//         gpuErrchk( cudaEventRecord(stopEvent, 0) );
//         gpuErrchk( cudaEventSynchronize(stopEvent) );
//         gpuErrchk( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
//         //std::cout << "number of blocks: " << num_blocks << std::endl;
//     	printf("Time for kernel launch (ms): %f\n", ms);

//     	//gpuErrchk( cudaMemcpyAsync(h_arr1, d_arr1, array_bytes, cudaMemcpyDeviceToHost) ); //transfer the ouput array to the CPU
//         gpuErrchk( cudaMemcpyAsync(h_arr1, d_result, array_bytes, cudaMemcpyDeviceToHost) ); //transfer the ouput array to the CPU
//         gpuErrchk( cudaDeviceSynchronize() );

//     	std::cout<<"Output array:"<<std::endl;
//     	for(uint64_t i = 0; i < array_size; i++){
//     		std::cout<<h_arr1[i]<< " ";
//     	}
//     	std::cout << std::endl;
//     //}
//  	return 0;
// }

//testing IO
// int main(void)
// {
//     using ps1 = gpu<v2048<uint64_t>>;
//     IMPORT_VECTOR_BOILER_PLATE(ps1)
//     const uint64_t array_size = 64;
//     const uint64_t array_bytes = array_size * sizeof(uint64_t);
//     const uint64_t array_32elements_bytes = 32 * sizeof(uint64_t);
//     uint64_t *h_arr1, *h_result, *d_arr1, *d_result, *h_result2;
//     gpuErrchk( cudaMallocHost((void**)&h_arr1, array_bytes) );
//     gpuErrchk( cudaMallocHost((void**)&h_result, array_32elements_bytes) );
//     gpuErrchk( cudaMallocHost((void**)&h_result2, array_bytes) );
//     gpuErrchk( cudaMalloc((void**)&d_arr1, array_bytes) );
//     gpuErrchk( cudaMalloc((void**)&d_result, array_32elements_bytes) );

//     float ms;
//     cudaEvent_t startEvent, stopEvent;
//     gpuErrchk( cudaEventCreate(&startEvent) );
//     gpuErrchk( cudaEventCreate(&stopEvent) );

//     //generate input array
//     srand(time(NULL));
//     for(uint64_t i = 0; i < array_size; i++){
//         h_arr1[i] = rand() % 10;//range 0 to 9
//     }
//     std::cout<<"h_arr1:"<<std::endl;
//     for(uint64_t i = 0; i < array_size; i++){
//         std::cout<<h_arr1[i]<< " ";
//     }
//     std::cout << std::endl;

//     gpuErrchk( cudaMemcpyAsync(d_arr1, h_arr1, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
//     gpuErrchk( cudaEventRecord(startEvent,0) );
//     d_result = load<ps1, iov::ALIGNED, vector_size_bit::value>(d_arr1);
//     gpuErrchk( cudaEventRecord(stopEvent, 0) );
//     gpuErrchk( cudaEventSynchronize(stopEvent) );
//     gpuErrchk( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
//     printf("Time for kernel launch (ms): %f\n", ms);
//     gpuErrchk( cudaMemcpyAsync(h_result, d_result, array_32elements_bytes, cudaMemcpyDeviceToHost) ); //transfer the ouput array to the CPU
//     gpuErrchk( cudaDeviceSynchronize() );
//     std::cout<<"Load 32 elements:"<<std::endl;
//     for(uint64_t i = 0; i < 32; i++){
//         std::cout<<h_result[i]<< " ";
//     }
//     std::cout << std::endl;
//     store<ps1, iov::ALIGNED, vector_size_bit::value>(d_arr1+32, d_result);
//     gpuErrchk( cudaMemcpyAsync(h_result2, d_arr1, array_bytes, cudaMemcpyDeviceToHost) ); //transfer the ouput array to the CPU
//     gpuErrchk( cudaDeviceSynchronize() );
//     std::cout<<"Stored array:"<<std::endl;
//     for(uint64_t i = 0; i < array_size; i++){
//         std::cout<<h_result2[i]<< " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }

//testing gather
// int main(void)
// {
//     using ps1 = gpu<v2048<uint64_t>>;
//     IMPORT_VECTOR_BOILER_PLATE(ps1)
//     const uint64_t array_size = 64;
//     const uint64_t array_bytes = array_size * sizeof(uint64_t);
//     const uint64_t array_32elements_bytes = 32 * sizeof(uint64_t);
//     uint64_t *h_arr1, *h_indices, *h_result, *d_arr1, *d_indices, *d_result;
//     gpuErrchk( cudaMallocHost((void**)&h_arr1, array_bytes) );
//     gpuErrchk( cudaMallocHost((void**)&h_indices, array_32elements_bytes) );
//     gpuErrchk( cudaMallocHost((void**)&h_result, array_32elements_bytes) );
//     gpuErrchk( cudaMalloc((void**)&d_arr1, array_bytes) );
//     gpuErrchk( cudaMalloc((void**)&d_indices, array_32elements_bytes) );
//     gpuErrchk( cudaMalloc((void**)&d_result, array_32elements_bytes) );

//     float ms;
//     cudaEvent_t startEvent, stopEvent;
//     gpuErrchk( cudaEventCreate(&startEvent) );
//     gpuErrchk( cudaEventCreate(&stopEvent) );

//     //generate input array
//     srand(time(NULL));
//     for(uint64_t i = 0; i < array_size; i++){
//         h_arr1[i] = rand() % 10;//range 0 to 9
//     }
//     for(uint64_t i = 0; i < 32; i++){
//         h_indices[i] = rand() % 5;//range 0 to 9
//     }
//     std::cout<<"h_arr1:"<<std::endl;
//     for(uint64_t i = 0; i < array_size; i++){
//         std::cout<<h_arr1[i]<< " ";
//     }
//     std::cout<<"h_indices:"<<std::endl;
//     for(uint64_t i = 0; i < 32; i++){
//         std::cout<<h_indices[i]<< " ";
//     }
//     std::cout << std::endl;

//     gpuErrchk( cudaMemcpyAsync(d_arr1, h_arr1, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
//     gpuErrchk( cudaMemcpyAsync(d_indices, h_indices, array_32elements_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU

//     gpuErrchk( cudaEventRecord(startEvent,0) );
//     d_result = vectorlib::gather<ps1, vector_base_t_granularity::value, 1>(d_arr1, d_indices);
//     gpuErrchk( cudaEventRecord(stopEvent, 0) );
//     gpuErrchk( cudaEventSynchronize(stopEvent) );
//     gpuErrchk( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
//     printf("Time for kernel launch (ms): %f\n", ms);
//     gpuErrchk( cudaMemcpyAsync(h_result, d_result, array_32elements_bytes, cudaMemcpyDeviceToHost) ); //transfer the ouput array to the CPU
//     gpuErrchk( cudaDeviceSynchronize() );
//     std::cout<<"Load 32 elements:"<<std::endl;
//     for(uint64_t i = 0; i < 32; i++){
//         std::cout<<h_result[i]<< " ";
//     }

//     return 0;
// }


//testing compressstore
// int main(void)
// {
//     using ps1 = gpu<v2048<uint64_t>>;
//     IMPORT_VECTOR_BOILER_PLATE(ps1)
//     const uint64_t array_size = 32;
//     const uint64_t array_bytes = array_size * sizeof(uint64_t);
//     uint64_t *h_arr1, *h_result, *d_arr1, *d_arr2, *h_arr2;;

//     gpuErrchk( cudaMallocHost((void**)&h_arr1, array_bytes) );
//     gpuErrchk( cudaMallocHost((void**)&h_arr2, array_bytes) );
//     gpuErrchk( cudaMallocHost((void**)&h_result, array_bytes) );
//     gpuErrchk( cudaMalloc((void**)&d_arr1, array_bytes) );
//     gpuErrchk( cudaMalloc((void**)&d_arr2, array_bytes) );
//     float ms;
//     cudaEvent_t startEvent, stopEvent;
//     gpuErrchk( cudaEventCreate(&startEvent) );
//     gpuErrchk( cudaEventCreate(&stopEvent) );

//     //generate input array
//     srand(time(NULL));
//     for(uint64_t i = 0; i < array_size; i++){
//         h_arr1[i] = rand() % 10;//range 0 to 9
//         h_arr2[i] = 0;
//     }
//     std::cout<<"h_arr1:"<<std::endl;
//     for(uint64_t i = 0; i < array_size; i++){
//         std::cout<<h_arr1[i]<< " ";
//     }
//     std::cout << std::endl;

//     gpuErrchk( cudaMemcpyAsync(d_arr1, h_arr1, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
//     gpuErrchk( cudaMemcpyAsync(d_arr2, h_arr2, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
//     gpuErrchk( cudaEventRecord(startEvent,0) );
//     compressstore<ps1, iov::UNALIGNED, vector_base_t_granularity::value>(d_arr2, d_arr1, 29);
//     gpuErrchk( cudaEventRecord(stopEvent, 0) );
//     gpuErrchk( cudaEventSynchronize(stopEvent) );
//     gpuErrchk( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
//     printf("Time for kernel launch (ms): %f\n", ms);
//     gpuErrchk( cudaMemcpyAsync(h_result, d_arr2, array_bytes, cudaMemcpyDeviceToHost) ); //transfer the ouput array to the CPU
//     gpuErrchk( cudaDeviceSynchronize() );
//     std::cout<<"Load 32 elements:"<<std::endl;
//     for(uint64_t i = 0; i < 32; i++){
//         std::cout<<h_result[i]<< " ";
//     }

//     return 0;
// }

//testing set
// int main(void)
// {
//     using ps1 = gpu<v2048<uint64_t>>;
//     const uint64_t array_size = 32;
//     uint64_t *h_arr, *d_arr;
//     h_arr = (uint64_t*)malloc(array_size*sizeof(uint64_t));
//     gpuErrchk( cudaMalloc((void**)&d_arr, array_size * sizeof(uint64_t)) );
//     float ms;
//     cudaEvent_t startEvent, stopEvent;
//     gpuErrchk( cudaEventCreate(&startEvent) );
//     gpuErrchk( cudaEventCreate(&stopEvent) );
//     gpuErrchk( cudaEventRecord(startEvent,0) );
//     //d_arr = vectorlib::set1<ps1, ps1::vector_helper_t::granularity::value>(5);
//     //d_arr = vectorlib::set_sequence<ps1, ps1::vector_helper_t::granularity::value>(0, 2);
//     d_arr = vectorlib::set<ps1, ps1::vector_helper_t::granularity::value>(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
//     gpuErrchk( cudaEventRecord(stopEvent, 0) );
//     gpuErrchk( cudaEventSynchronize(stopEvent) );
//     gpuErrchk( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
//     printf("Time for kernel launch (ms): %f\n", ms);
//     gpuErrchk( cudaMemcpy(h_arr, d_arr, array_size*sizeof(uint64_t), cudaMemcpyDeviceToHost) ); //transfer the ouput array to the CPU
//     gpuErrchk( cudaDeviceSynchronize() );
//     for(uint64_t i = 0; i < array_size; i++){
//         std::cout<<h_arr[i]<< " ";
//     }
//     std::cout << std::endl;
//     return 0;
// }

//testing extract
// int main(void)
// {
//     using ps1 = gpu<v2048<uint64_t>>;
//     const uint64_t array_size = 32;
//     const uint64_t array_bytes = array_size * sizeof(uint64_t);
//     uint64_t h_arr[array_size];
//     uint64_t h_result = 0;
//     uint64_t *d_result, *d_arr;
//     gpuErrchk( cudaMalloc((void**)&d_arr, array_bytes) );
//     gpuErrchk( cudaMalloc((void**)&d_result, sizeof(uint64_t)) );
//     srand(time(NULL));
//     for(uint64_t i = 0; i < array_size; i++){
//         h_arr[i] = rand() % 10;//range 0 to 9
//     }
//     std::cout<<"h_arr:"<<std::endl;
//     for(uint64_t i = 0; i < array_size; i++){
//         std::cout<<h_arr[i]<< " ";
//     }
//     std::cout << std::endl;
//     gpuErrchk( cudaMemcpy(d_arr, h_arr, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
//     for(int i=0; i < array_size; i++){
//         d_result = extract_value<ps1,ps1::vector_helper_t::granularity::value>(d_arr, i);
//         gpuErrchk( cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost) ); //transfer the ouput array to the CPU
//         std::cout << h_result << " ";
//     }
//     std::cout << std::endl;
// }


// //testing hadd
// int main(void)
// {
// 	using ps1 = gpu<v2048<uint64_t>>;
// 	const uint64_t array_size = 32;
// 	const uint64_t array_bytes = array_size * sizeof(uint64_t);
// 	uint64_t *h_arr1;
// 	uint64_t *d_arr1;
//     uint64_t h_result = 0;
//     uint64_t *d_result;
//     gpuErrchk( cudaMalloc((void**)&d_result, sizeof(uint64_t)) );

// 	gpuErrchk( cudaMallocHost((void**)&h_arr1, array_bytes) );
// 	gpuErrchk( cudaMalloc((void**)&d_arr1, array_bytes) );
// 	float ms;
//     cudaEvent_t startEvent, stopEvent;
//     gpuErrchk( cudaEventCreate(&startEvent) );
//     gpuErrchk( cudaEventCreate(&stopEvent) );
// 	srand(time(NULL));
// 	for(uint64_t i = 0; i < array_size; i++){
// 		h_arr1[i] = rand() % 10;//range 0 to 9
// 	}
// 	std::cout<<"h_arr1:"<<std::endl;
// 	for(uint64_t i = 0; i < array_size; i++){
// 		std::cout<<h_arr1[i]<< " ";
// 	}
// 	std::cout << std::endl;
//     gpuErrchk( cudaMemcpyAsync(d_arr1, h_arr1, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
// 	gpuErrchk( cudaEventRecord(startEvent,0) );
// 	d_result = hadd<ps1, ps1::vector_helper_t::granularity::value>::apply(d_arr1);
// 	gpuErrchk( cudaEventRecord(stopEvent, 0) );
//     gpuErrchk( cudaEventSynchronize(stopEvent) );
//     gpuErrchk( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
// 	printf("Time for kernel launch (ms): %f\n", ms);
// 	gpuErrchk( cudaMemcpyAsync(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost) );
// 	gpuErrchk( cudaDeviceSynchronize() );
// 	std::cout<<"Output: "<< h_result << std::endl;
//  	return 0;
// }

//testing rotate
// int main(void)
// {
// 	using ps1 = gpu<v2048<uint64_t>>;
// 	const uint64_t array_size = 32;
// 	const uint64_t array_bytes = array_size * sizeof(uint64_t);
// 	uint64_t *h_arr1;
// 	uint64_t *d_arr1;
// 	gpuErrchk( cudaMallocHost((void**)&h_arr1, array_bytes) );
// 	gpuErrchk( cudaMalloc((void**)&d_arr1, array_bytes) );
// 	float ms;
//     cudaEvent_t startEvent, stopEvent;
//     gpuErrchk( cudaEventCreate(&startEvent) );
//     gpuErrchk( cudaEventCreate(&stopEvent) );
// 	srand(time(NULL));
// 	for(uint64_t i = 0; i < array_size; i++){
// 		h_arr1[i] = rand() % 10;//range 0 to 9
// 	}
// 	std::cout<<"h_arr1:"<<std::endl;
// 	for(uint64_t i = 0; i < array_size; i++){
// 		std::cout<<h_arr1[i]<< " ";
// 	}
// 	std::cout << std::endl;
//     gpuErrchk( cudaMemcpyAsync(d_arr1, h_arr1, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
// 	gpuErrchk( cudaEventRecord(startEvent,0) );
// 	manipulate<ps1, ps1::vector_helper_t::granularity::value>::rotate(d_arr1);
// 	gpuErrchk( cudaEventRecord(stopEvent, 0) );
//     gpuErrchk( cudaEventSynchronize(stopEvent) );
//     gpuErrchk( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
// 	printf("Time for kernel launch (ms): %f\n", ms);
// 	gpuErrchk( cudaMemcpyAsync(h_arr1, d_arr1, array_bytes, cudaMemcpyDeviceToHost) ); //todo: ideally only d_arr[0] needs to be copied, not the whole vector
// 	gpuErrchk( cudaDeviceSynchronize() );
// 	std::cout<<"Output array:"<<std::endl;
// 	for(uint64_t i = 0; i < array_size; i++){
// 		std::cout<<h_arr1[i]<< " ";
// 	}	std::cout << std::endl;
//  	return 0;
// }

//testing shift
// int main(void)
// {
// 	using ps1 = gpu<v2048<uint64_t>>;
// 	const uint64_t array_size = 32;
// 	const uint64_t array_bytes = array_size * sizeof(uint64_t);
// 	uint64_t *h_arr1;
// 	uint64_t *d_arr1;
// 	gpuErrchk( cudaMallocHost((void**)&h_arr1, array_bytes) );
// 	gpuErrchk( cudaMalloc((void**)&d_arr1, array_bytes) );
// 	float ms;
//     cudaEvent_t startEvent, stopEvent;
//     gpuErrchk( cudaEventCreate(&startEvent) );
//     gpuErrchk( cudaEventCreate(&stopEvent) );
// 	srand(time(NULL));
// 	for(uint64_t i = 0; i < array_size; i++){
// 		h_arr1[i] = rand() % 10;//range 0 to 9
// 	}
// 	std::cout<<"h_arr1:"<<std::endl;
// 	for(uint64_t i = 0; i < array_size; i++){
// 		std::cout<<h_arr1[i]<< " ";
// 	}
// 	int distance = 1;
// 	std::cout << std::endl;
//     gpuErrchk( cudaMemcpyAsync(d_arr1, h_arr1, array_bytes, cudaMemcpyHostToDevice) ); //transfer the input array to the GPU
// 	gpuErrchk( cudaEventRecord(startEvent,0) );
// 	shift_right<ps1, ps1::vector_helper_t::granularity::value>::apply(d_arr1, distance);
// 	gpuErrchk( cudaEventRecord(stopEvent, 0) );
//     gpuErrchk( cudaEventSynchronize(stopEvent) );
//     gpuErrchk( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
// 	printf("Time for kernel launch (ms): %f\n", ms);
// 	gpuErrchk( cudaMemcpyAsync(h_arr1, d_arr1, array_bytes, cudaMemcpyDeviceToHost) );
// 	gpuErrchk( cudaDeviceSynchronize() );
// 	// std::cout<<"Output: "<< h_arr1[0] << std::endl;
// 	// std::cout << std::endl;
// 	for(uint64_t i = 0; i < array_size; i++){
// 		std::cout<<h_arr1[i]<< " ";
// 	}	std::cout << std::endl;

//  	return 0;
// }