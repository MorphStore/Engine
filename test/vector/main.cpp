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
   
   const column< uncompr_f > * testDataColumnSorted = generate_sorted_unique(100,1,5);
   const uint64_t* data = (uint64_t*) testDataColumnSorted->get_data();
   
   auto outColumn = new column<uncompr_f>(100);
   avx2< v256< uint64_t > >::vector_t gatherTest256 = _mm256_set_epi64x(0,0,0,1);
   sse< v128< uint64_t > >::vector_t gatherTest128 = _mm_set_epi64x(0,1);
   
   
   uint64_t temp=_mm_extract_epi64((load<sse< v128< uint64_t > >, iov::ALIGNED, 128>(data)),0);
   std::cout << "sse aligned " << temp << "\n";
   
   temp=_mm_extract_epi64((load<sse< v128< uint64_t > >, iov::UNALIGNED, 128>(data)),0);
   std::cout << "sse unaligned " << temp << "\n";
   
   sse< v128< uint64_t > >::vector_t testvec128;
   testvec128=load<sse< v128< uint64_t > >, iov::UNALIGNED, 128>(data);
   vector::store < sse < v128 < uint64_t > > , iov::ALIGNED, 128 > ((uint64_t*) outColumn->get_data(),testvec128);
   std::cout << "sse aligned store " << ((uint64_t*) outColumn->get_data())[0] << "\n";
   
   compressstore< sse < v128 < uint64_t > > , iov::UNALIGNED, 128 > ((uint64_t*) outColumn->get_data(),testvec128,2);
   std::cout << "sse compress store, 128 bit " << ((uint64_t*) outColumn->get_data())[0] << "\n";
   
   testvec128=gather<sse< v128< uint64_t > >, iov::UNALIGNED, 128>(data,gatherTest128);
   temp=_mm_extract_epi64(testvec128,0);
   std::cout << "sse gather " << temp << "\n";
   
   
   temp=_mm256_extract_epi64((load<avx2< v256< uint64_t > >, iov::ALIGNED, 256>(data)),0);
   std::cout << "avx2 aligned " << temp << "\n";

   temp=_mm256_extract_epi64((load<avx2< v256< uint64_t > >, iov::UNALIGNED, 256>(data)),0);
   std::cout << "avx2 unaligned " << temp << "\n";
   
   avx2< v256< uint64_t > >::vector_t testvec256;
   testvec256=load<avx2< v256< uint64_t > >, iov::UNALIGNED, 256>(data);
   vector::store < avx2 < v256 < uint64_t > > , iov::ALIGNED, 256 > ((uint64_t*) outColumn->get_data(),testvec256);
   std::cout << "avx2 aligned store " << ((uint64_t*) outColumn->get_data())[0] << "\n";
   
   compressstore< avx2 < v256 < uint64_t > > , iov::UNALIGNED, 256 > ((uint64_t*) outColumn->get_data(),testvec256,2);
   std::cout << "avx2 compress store, 256 bit " << ((uint64_t*) outColumn->get_data())[0] << "\n";
   
   testvec128=gather<avx2< v128< uint64_t > >, iov::UNALIGNED, 128>(data,gatherTest128);
   temp=_mm_extract_epi64(testvec128,0);
   std::cout << "avx2 gather, 128 bit " << temp << "\n";
   
   testvec256=gather<avx2< v256< uint64_t > >, iov::UNALIGNED, 256>(data,gatherTest256);
   temp=_mm256_extract_epi64(testvec256,0);
   std::cout << "avx2 gather, 256 bit " << temp << "\n";
   
   
   
   #ifdef AVX512

   avx512< v512< uint64_t > >::vector_t gatherTest512 = _mm512_set_epi64(0,0,0,0,0,0,0,1);
   avx512< v512< uint64_t > >::vector_t testvec512;
   
   temp=_mm256_extract_epi64(_mm512_extracti64x4_epi64((load<avx512< v512< uint64_t > >, iov::ALIGNED, 512>(data)),0),1);
   std::cout << "avx512 aligned " << temp << "\n";
   
   temp=_mm256_extract_epi64(_mm512_extracti64x4_epi64((load<avx512< v512< uint64_t > >, iov::UNALIGNED, 512>(data)),0),1);
   std::cout << "avx512 unaligned " << temp << "\n";
   
   
   testvec512=load<avx512< v512< uint64_t > >, iov::UNALIGNED, 512>(data);
   vector::store < avx512 < v512 < uint64_t > > , iov::ALIGNED, 512 > ((uint64_t*) outColumn->get_data(),testvec512);
   std::cout << "avx512 aligned store " << ((uint64_t*) outColumn->get_data())[0] << "\n";
   
   vector::compressstore< avx512 < v512 < uint64_t > > , iov::UNALIGNED, 512 > ((uint64_t*) outColumn->get_data(),testvec512,128);
   std::cout << "avx512 compress store, 512 bit " << ((uint64_t*) outColumn->get_data())[0] << "\n";
   
   compressstore< avx512 < v256 < uint64_t > > , iov::UNALIGNED, 256 > ((uint64_t*) outColumn->get_data(),testvec256,4);
   std::cout << "avx512 compress store, 256 bit " << ((uint64_t*) outColumn->get_data())[0] << "\n";
   
   compressstore< avx512 < v128 < uint64_t > > , iov::UNALIGNED, 128 > ((uint64_t*) outColumn->get_data(),testvec128,2);
   std::cout << "avx512 compress store, 128 bit " << ((uint64_t*) outColumn->get_data())[0] << "\n";
   
   testvec512=gather<avx512< v512< uint64_t > >, iov::UNALIGNED, 512>(data,gatherTest512);
   temp=_mm256_extract_epi64(_mm512_extracti64x4_epi64(testvec512,0),0);
   std::cout << "avx512 gather " << temp << "\n";
   
   #endif
   
   return 0;
}