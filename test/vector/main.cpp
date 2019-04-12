/**
 * @file main.cpp
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#include <core/memory/mm_glob.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>

#include <vector/general_vector.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/avx2/primitives/io_avx2.h>

#ifdef AVX512
#include <vector/simd/avx512/primitives/io_avx512.h>
#endif

#include <iostream>
#include <immintrin.h>



int main( void ) {
    

   using namespace vector;
   using namespace morphstore;
   
   const column< uncompr_f > * testDataColumnSorted = generate_sorted_unique(100,1,1);
   const uint64_t* data = (uint64_t*) testDataColumnSorted->get_data();
   //const void * datav = (const void*) data; 
   

   int temp=_mm_extract_epi64((load<sse< v128< uint64_t > >, iov::ALIGNED, 128>(data)),0);
   std::cout << "sse aligned " << temp << "\n";
   
   temp=_mm_extract_epi64((load<sse< v128< uint64_t > >, iov::UNALIGNED, 128>(data)),0);
   std::cout << "sse unaligned " << temp << "\n";
   
   temp=_mm256_extract_epi64((load<avx2< v256< uint64_t > >, iov::ALIGNED, 256>(data)),0);
   std::cout << "avx2 aligned " << temp << "\n";

   temp=_mm256_extract_epi64((load<avx2< v256< uint64_t > >, iov::UNALIGNED, 256>(data)),0);
   std::cout << "avx2 unaligned " << temp << "\n";
   
   #ifdef AVX512
   temp=_mm256_extract_epi64(_mm512_extracti64x4_epi64((load<avx512< v512< uint64_t > >, iov::ALIGNED, 512>(data)),0),0);
   std::cout << "avx512 aligned " << temp << "\n";
   
   temp=_mm256_extract_epi64(_mm512_extracti64x4_epi64((load<avx512< v512< uint64_t > >, iov::UNALIGNED, 512>(data)),0),0);
   std::cout << "avx512 unaligned " << temp << "\n";
   #endif
   
   return 0;
}