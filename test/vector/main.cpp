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
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/compare_sse.h>
#include <vector/simd/sse/primitives/manipulate_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include <vector/simd/sse/primitives/extract_sse.h>

#ifdef AVXTWO
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>
#include <vector/simd/avx2/primitives/manipulate_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/extract_avx2.h>
#endif

#ifdef AVX512
#include <vector/simd/avx512/primitives/io_avx512.h>
#include <vector/simd/avx512/primitives/calc_avx512.h>
#include <vector/simd/avx512/primitives/compare_avx512.h>
#include <vector/simd/avx512/primitives/manipulate_avx512.h>
#include <vector/simd/avx512/primitives/create_avx512.h>
#include <vector/simd/avx512/primitives/extract_avx512.h>
#endif

#include <iostream>
#include <immintrin.h>



int main( void ) {
    

   using namespace vector;
   using namespace morphstore;
   
   const column< uncompr_f > * testDataColumnSorted = generate_sorted_unique(100,1,5);
   const uint64_t* data = (uint64_t*) testDataColumnSorted->get_data();
   
   auto outColumn = new column<uncompr_f>(100);
   
   sse< v128< uint64_t > >::vector_t gatherTest128 = _mm_set_epi64x(0,4);
   
   uint64_t  temp;
   int temp4;
   double temp2;
   float temp3;
   
   temp=extract_value<sse< v128< uint64_t > >, 64>((load<sse< v128< uint64_t > >, iov::ALIGNED, 128>(data)),0);
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
   
   temp=_mm_extract_epi64((add<sse< v128< uint64_t > >, 64>(testvec128,gatherTest128)),0);
   std::cout << "sse add 64 bit " << temp << "\n";
   
   temp=_mm_extract_epi32((add<sse< v128< uint32_t > >, 32>(testvec128,gatherTest128)),0);
   std::cout << "sse add 32 bit " << temp << "\n";
   
   temp=_mm_extract_epi64((sub<sse< v128< uint64_t > >, 64>(testvec128,gatherTest128)),0);
   std::cout << "sse sub 64 bit " << temp << "\n";
   
   temp=_mm_extract_epi32((sub<sse< v128< uint32_t > >, 32>(testvec128,gatherTest128)),0);
   std::cout << "sse sub 32 bit " << temp << "\n";
   
   temp=hadd<sse< v128< uint64_t > >, 64>(testvec128);
   std::cout << "sse hadd 64 bit " << temp << "\n";
   
   temp=hadd<sse< v128< uint32_t > >, 32>(testvec128);
   std::cout << "sse hadd 32 bit " << temp << "\n";
   
   temp3 = hadd<sse< v128< float > >, 32>((__m128)testvec128);
   std::cout << "sse hadd 32 bit (float) " << temp3 << "\n";
   
   temp=_mm_extract_epi64((mul<sse< v128< uint64_t > >, 64>(testvec128,gatherTest128)),0);
   std::cout << "sse mul 64 bit " << temp << "\n";
   
   temp=_mm_extract_epi32((mul<sse< v128< uint32_t > >, 32>(testvec128,gatherTest128)),0);
   std::cout << "sse mul 32 bit " << temp << "\n";
   
   temp=_mm_extract_epi64((div<sse< v128< uint64_t > >, 64>(testvec128,gatherTest128)),0);
   std::cout << "sse div 64 bit " << temp << "\n";
   
   temp2=_mm_extract_epi64((__m128i)(div<sse< v128< double > >, 64>((__m128d)testvec128,(__m128d)gatherTest128)),0);
   std::cout << "sse div 64 bit (double) " << temp2 << "\n";
   
   temp3=_mm_extract_ps((div<sse< v128< float > >, 32>((__m128)testvec128,(__m128)gatherTest128)),0);
   std::cout << "sse div 32 bit (float) " << temp3 << "\n";
   
   temp=_mm_extract_epi64((mod<sse< v128< uint64_t > >, 64>(testvec128,gatherTest128)),0);
   std::cout << "sse mod 64 bit " << _mm_extract_epi64(testvec128,0) << " " << _mm_extract_epi64(gatherTest128,0) << ": " << temp << "\n";
   
   
   temp4=_mm_extract_epi64((inv<sse< v128< uint64_t > >, 64>(testvec128)),0);
   std::cout << "sse inv 64 bit " << temp4 << "\n";
   
   temp2=_mm_extract_epi64((__m128i)(inv<sse< v128< double > >, 64>((__m128d)testvec128)),0);
   std::cout << "sse inv 64 bit (double) " << temp2 << "\n";
   
   temp4=_mm_extract_epi32((inv<sse< v128< uint64_t > >, 32>(testvec128)),0);
   std::cout << "sse inv 32 bit " << temp4 << "\n";
   
   temp3=_mm_extract_ps((inv<sse< v128< float > >, 32>((__m128)testvec128)),0);
   std::cout << "sse inv 32 bit (float) " << temp3 << "\n";
   
   temp=equality<sse< v128< uint64_t > >, 64>(testvec128,testvec128);
   std::cout << "sse equality 64 bit " << temp << "\n";
   
   temp=equality<sse< v128< uint64_t > >, 32>(testvec128,testvec128);
   std::cout << "sse equality 32 bit " << temp << "\n";
   
   temp=lessthan<sse< v128< uint64_t > >, 64>(testvec128,testvec128);
   std::cout << "sse less than 64 bit " << temp << "\n";
   
   temp=lessthan<sse< v128< uint64_t > >, 32>(testvec128,testvec128);
   std::cout << "sse less than 32 bit " << temp << "\n";
   
   temp=greaterthan<sse< v128< uint64_t > >, 64>(testvec128,testvec128);
   std::cout << "sse greater than 64 bit " << temp << "\n";
   
   temp=greaterthan<sse< v128< uint64_t > >, 32>(testvec128,testvec128);
   std::cout << "sse greater than 32 bit " << temp << "\n";
   
   temp=greaterequal<sse< v128< uint64_t > >, 64>(testvec128,testvec128);
   std::cout << "sse greater equal 64 bit " << temp << "\n";
   
   temp=greaterequal<sse< v128< uint64_t > >, 32>(testvec128,testvec128);
   std::cout << "sse greater equal 32 bit " << temp << "\n";
   
   temp=lessequal<sse< v128< uint64_t > >, 64>(testvec128,testvec128);
   std::cout << "sse less equal 64 bit " << temp << "\n";
   
   temp=lessequal<sse< v128< uint64_t > >, 32>(testvec128,testvec128);
   std::cout << "sse less equal 32 bit " << temp << "\n";
   
   temp=_mm_extract_epi64((rotate<sse< v128< uint64_t > >, 64>(testvec128)),0);
   std::cout << "sse rotate 64 bit " << temp << "\n";
   
   temp=_mm_extract_epi64((set1<sse< v128< uint64_t > >, 64>(42)),0);
   std::cout << "sse set1 64 bit " << temp << "\n";
   
   temp=_mm_extract_epi64((set<sse< v128< uint64_t > >, 64>(44,43)),0);
   std::cout << "sse set 64 bit " << temp << "\n";
   
   temp=_mm_extract_epi32((set1<sse< v128< uint64_t > >, 32>(42)),0);
   std::cout << "sse set1 32 bit " << temp << "\n";
   
   temp=_mm_extract_epi32((set<sse< v128< uint64_t > >, 32>(46,45,44,43)),0);
   std::cout << "sse set 32 bit " << temp << "\n";
   
   temp=_mm_extract_epi64((set_sequence<sse< v128< uint64_t > >, 64>(0,5)),1);
   std::cout << "sse set_sequence 64 bit " << temp << "\n";
   
   temp=_mm_extract_epi32((set_sequence<sse< v128< uint64_t > >, 32>(0,5)),1);
   std::cout << "sse set_sequence 32 bit " << temp << "\n";
   
   #ifdef AVXTWO

   avx2< v256< uint64_t > >::vector_t gatherTest256 = _mm256_set_epi64x(1,1,1,2);

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
   
   temp=_mm256_extract_epi64((add<avx2< v256< uint64_t > >, 64>(testvec256,gatherTest256)),0);
   std::cout << "avx2 add 64 bit " << temp << "\n";
   
   temp=_mm256_extract_epi32((add<avx2< v256< uint32_t > >, 32>(testvec256,gatherTest256)),0);
   std::cout << "avx2 add 32 bit " << temp << "\n";
   
   temp=_mm256_extract_epi64((sub<avx2< v256< uint64_t > >, 64>(testvec256,gatherTest256)),0);
   std::cout << "avx2 sub 64 bit " << temp << "\n";
   
   temp=_mm256_extract_epi32((sub<avx2< v256< uint32_t > >, 32>(testvec256,gatherTest256)),0);
   std::cout << "avx2 sub 32 bit " << temp << "\n";
   
   temp=hadd<avx2< v256< uint64_t > >, 64>(testvec256);
   std::cout << "avx2 hadd 64 bit " << temp << "\n";
   
   temp=hadd<avx2< v256< uint32_t > >, 32>(testvec256);
   std::cout << "avx2 hadd 32 bit " << temp << "\n";
   
   temp=_mm256_extract_epi64((mul<avx2< v256< uint64_t > >, 64>(testvec256,gatherTest256)),0);
   std::cout << "avx2 mul 64 bit " << temp << "\n";
   
   temp=_mm256_extract_epi32((mul<avx2< v256< uint32_t > >, 32>(testvec256,gatherTest256)),0);
   std::cout << "avx2 mul 32 bit " << temp << "\n";
   
   temp=_mm256_extract_epi64((div<avx2< v256< uint64_t > >, 64>(testvec256,gatherTest256)),0);
   std::cout << "avx2 div 64 bit " << temp << "\n";
   
   temp2=_mm256_extract_epi64((div<avx2< v256< double > >, 64>((__m256d)testvec256,(__m256d)gatherTest256)),0);
   std::cout << "avx2 div 64 bit (double) " << temp2 << "\n";
   
   temp3=_mm256_extract_epi32((div<avx2< v256< float > >, 32>((__m256)testvec256,(__m256)gatherTest256)),0);
   std::cout << "avx2 div 32 bit (float) " << temp3 << "\n";
   
   temp=_mm256_extract_epi64((mod<avx2< v256< uint64_t > >, 64>(testvec256,gatherTest256)),0);
   std::cout << "avx2 mod 64 bit " << _mm256_extract_epi64(testvec256,0) << " " << _mm256_extract_epi64(gatherTest256,0) << ": " << temp << "\n";
   
   temp4=_mm256_extract_epi64((inv<avx2< v256< uint64_t > >, 64>(testvec256)),0);
   std::cout << "avx2 inv 64 bit " << temp4 << "\n";
   
   temp2=_mm256_extract_epi64((inv<avx2< v256< double > >, 64>((__m256d)testvec256)),0);
   std::cout << "avx2 inv 64 bit (double) " << temp2 << "\n";
   
   temp4=_mm256_extract_epi32((inv<avx2< v256< uint64_t > >, 32>(testvec256)),0);
   std::cout << "avx2 inv 32 bit " << temp4 << "\n";
   
   temp3=_mm256_extract_epi32((inv<avx2< v256< float > >, 32>((__m256)testvec256)),0);
   std::cout << "avx2 inv 32 bit (float) " << temp3 << "\n";
   
   temp=equality<avx2< v256< uint64_t > >, 64>(testvec256,testvec256);
   std::cout << "avx2 equality 64 bit " << temp << "\n";
   
   temp=equality<avx2< v256< uint64_t > >, 32>(testvec256,testvec256);
   std::cout << "avx2 equality 32 bit " << temp << "\n";
   
   temp=lessthan<avx2< v256< uint64_t > >, 64>(testvec256,testvec256);
   std::cout << "avx2 less than 64 bit " << temp << "\n";
   
   temp=lessthan<avx2< v256< uint64_t > >, 32>(testvec256,testvec256);
   std::cout << "avx2 less than 32 bit " << temp << "\n";
   
   temp=greaterthan<avx2< v256< uint64_t > >, 64>(testvec256,testvec256);
   std::cout << "avx2 greater than 64 bit " << temp << "\n";
   
   temp=greaterthan<avx2< v256< uint64_t > >, 32>(testvec256,testvec256);
   std::cout << "avx2 greater than 32 bit " << temp << "\n";
   
   temp=greaterequal<avx2< v256< uint64_t > >, 64>(testvec256,testvec256);
   std::cout << "avx2 greater equal 64 bit " << temp << "\n";
   
   temp=greaterequal<avx2< v256< uint64_t > >, 32>(testvec256,testvec256);
   std::cout << "avx2 greater equal 32 bit " << temp << "\n";
   
   temp=lessequal<avx2< v256< uint64_t > >, 64>(testvec256,testvec256);
   std::cout << "avx2 less equal 64 bit " << temp << "\n";
   
   temp=lessequal<avx2< v256< uint64_t > >, 32>(testvec256,testvec256);
   std::cout << "avx2 less equal 32 bit " << temp << "\n";
   
   temp=_mm256_extract_epi64((rotate<avx2< v256< uint64_t > >, 64>(testvec256)),0);
   std::cout << "avx2 rotate 64 bit " << temp << "\n";
   
   temp=_mm256_extract_epi64((set1<avx2< v256< uint64_t > >, 64>(42)),0);
   std::cout << "avx2 set1 64 bit " << temp << "\n";
   
   temp=_mm256_extract_epi64((set<avx2< v256< uint64_t > >, 64>(46,45,44,43)),0);
   std::cout << "avx2 set 64 bit " << temp << "\n";
   
   temp=_mm256_extract_epi32((set1<avx2< v256< uint64_t > >, 32>(42)),0);
   std::cout << "avx2 set1 32 bit " << temp << "\n";
   
   temp=_mm256_extract_epi32((set<avx2< v256< uint64_t > >, 32>(50,49,48,47,46,45,44,43)),0);
   std::cout << "avx2 set 32 bit " << temp << "\n";
   
   temp=_mm256_extract_epi64((set_sequence<avx2< v256< uint64_t > >, 64>(0,5)),3);
   std::cout << "avx2 set_sequence 64 bit " << temp << "\n";
   
   temp=extract_value<avx2< v256< uint64_t > >, 32>((set_sequence<avx2< v256< uint64_t > >, 32>(0,5)),3);
   std::cout << "avx2 set_sequence 32 bit " << temp << "\n";
   

   
   #endif

   
   #ifdef AVX512

   avx512< v512< uint64_t > >::vector_t gatherTest512 = _mm512_set_epi64(0,0,0,0,0,0,0,2);
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
   temp=extract_value<avx512< v512< uint64_t > >, 64>(testvec512,0);
   std::cout << "avx512 gather " << temp << "\n";
   
   temp=extract_value<avx512< v512< uint64_t > >, 64>((add<avx512< v512< uint64_t > >, 64>(testvec512,gatherTest512)),0);
   std::cout << "avx512 add 64 bit " << temp << "\n";
   
   temp=extract_value<avx512< v512< uint64_t > >, 32>((add<avx512< v512< uint32_t > >, 32>(testvec512,gatherTest512)),0);
   std::cout << "avx512 add 32 bit " << temp << "\n";
   
   temp=extract_value<avx512< v512< uint64_t > >, 64>((sub<avx512< v512< uint64_t > >, 64>(testvec512,gatherTest512)),0);
   std::cout << "avx512 sub 64 bit " << temp << "\n";
   
   temp=extract_value<avx512< v512< uint64_t > >, 32>((sub<avx512< v512< uint32_t > >, 32>(testvec512,gatherTest512)),0);
   std::cout << "avx512 sub 32 bit " << temp << "\n";
   
   temp=hadd<avx512< v512< uint64_t > >, 64>(testvec512);
   std::cout << "avx512 hadd 64 bit " << temp << "\n";
   
   temp2=hadd<avx512< v512< double > >, 64>((__m512d) testvec512);
   std::cout << "avx512 hadd 64 bit (double) " << temp2 << "\n";
   
   temp=hadd<avx512< v512< uint32_t > >, 32>(testvec512);
   std::cout << "avx512 hadd 32 bit " << temp << "\n";
   
   temp3=hadd<avx512< v512< float > >, 32>((__m512) testvec512);
   std::cout << "avx512 hadd 32 bit (float) " << temp3 << "\n";
   
   temp=extract_value<avx512< v512< uint64_t > >, 64>((mul<avx512< v512< uint64_t > >, 64>(testvec512,gatherTest512)),0);
   std::cout << "avx512 mul 64 bit " << temp << "\n";
   
   temp=extract_value<avx512< v512< uint64_t > >, 32>((mul<avx512< v512< uint32_t > >, 32>(testvec512,gatherTest512)),0);
   std::cout << "avx512 mul 32 bit " << temp << "\n";
   
   temp=extract_value<avx512< v512< uint64_t > >, 64>((div<avx512< v512< uint64_t > >, 64>(testvec512,gatherTest512)),0);
   std::cout << "avx512 div 64 bit " << temp << "\n";
   
   temp3=extract_value<avx512< v512< uint64_t > >, 64>((div<avx512< v512< double > >, 64>((__m512d)testvec512,(__m512d)gatherTest512)),0);
   std::cout << "avx512 div 64 bit (double) " << temp << "\n";
   
   temp2=extract_value<avx512< v512< uint64_t > >, 32>((div<avx512< v512< float > >, 32>((__m512)testvec512,(__m512)gatherTest512)),0);
   std::cout << "avx512 div 32 bit (float) " << temp << "\n";
   
   temp=extract_value<avx512< v512< uint64_t > >, 64>((mod<avx512< v512< uint64_t > >, 64>(testvec512,gatherTest512)),0);
   std::cout << "avx512 mod 64 bit " << temp << "\n";
   
   temp4=extract_value<avx512< v512< uint64_t > >, 64>((inv<avx512< v512< uint64_t > >, 64>(testvec512)),0);
   std::cout << "avx512 inv 64 bit " << temp4 << "\n";
   
   temp4=extract_value<avx512< v512< uint64_t > >, 32>((inv<avx512< v512< uint64_t > >, 32>(testvec512)),0);
   std::cout << "avx512 inv 32 bit " << temp4 << "\n";
   
   temp3=extract_value<avx512< v512< uint64_t > >, 32>((inv<avx512< v512< double > >, 64>((__m512d)testvec512)),0);
   std::cout << "avx512 inv 64 bit (double) " << temp3 << "\n";
   
   temp2=extract_value<avx512< v512< uint64_t > >, 32>((inv<avx512< v512< float > >, 32>((__m512)testvec512)),0);
   std::cout << "avx512 inv 32 bit (float) " << temp2 << "\n";
   
   temp=equality<avx512< v512< uint64_t > >, 64>(testvec512,testvec512);
   std::cout << "avx512 equality 64 bit " << temp << "\n";
   
   temp=equality<avx512< v512< uint64_t > >, 32>(testvec512,testvec512);
   std::cout << "avx512 equality 32 bit " << temp << "\n";
   
   temp=lessthan<avx512< v512< uint64_t > >, 64>(testvec512,testvec512);
   std::cout << "avx512 less than 64 bit " << temp << "\n";
   
   temp=lessthan<avx512< v512< uint64_t > >, 32>(testvec512,testvec512);
   std::cout << "avx512 less than 32 bit " << temp << "\n";
   
   temp=greaterthan<avx512< v512< uint64_t > >, 64>(testvec512,testvec512);
   std::cout << "avx512 greater than 64 bit " << temp << "\n";
   
   temp=greaterthan<avx512< v512< uint64_t > >, 32>(testvec512,testvec512);
   std::cout << "avx512 greater than 32 bit " << temp << "\n";
   
   temp=greaterequal<avx512< v512< uint64_t > >, 64>(testvec512,testvec512);
   std::cout << "avx512 greater equal 64 bit " << temp << "\n";
   
   temp=greaterequal<avx512< v512< uint64_t > >, 32>(testvec512,testvec512);
   std::cout << "avx512 greater equal 32 bit " << temp << "\n";
   
   temp=lessequal<avx512< v512< uint64_t > >, 64>(testvec512,testvec512);
   std::cout << "avx512 less equal 64 bit " << temp << "\n";
   
   temp=lessequal<avx512< v512< uint64_t > >, 32>(testvec512,testvec512);
   std::cout << "avx512 less equal 32 bit " << temp << "\n";
   
   temp=equality<avx512< v256< uint64_t > >, 64>(testvec256,testvec256);
   std::cout << "avx512 equality 64 bit (v256) " << temp << "\n";
   
   temp=equality<avx512< v256< uint64_t > >, 32>(testvec256,testvec256);
   std::cout << "avx512 equality 32 bit (v256) " << temp << "\n";
   
   temp=lessthan<avx512< v256< uint64_t > >, 64>(testvec256,testvec256);
   std::cout << "avx512 less than 64 bit (v256) " << temp << "\n";
   
   temp=lessthan<avx512< v256< uint64_t > >, 32>(testvec256,testvec256);
   std::cout << "avx512 less than 32 bit (v256) " << temp << "\n";
   
   temp=greaterthan<avx512< v256< uint64_t > >, 64>(testvec256,testvec256);
   std::cout << "avx512 greater than 64 bit (v256) " << temp << "\n";
   
   temp=greaterthan<avx512< v256< uint64_t > >, 32>(testvec256,testvec256);
   std::cout << "avx512 greater than 32 bit (v256) " << temp << "\n";
   
   temp=greaterequal<avx512< v256< uint64_t > >, 64>(testvec256,testvec256);
   std::cout << "avx512 greater equal 64 bit (v256) " << temp << "\n";
   
   temp=greaterequal<avx512< v256< uint64_t > >, 32>(testvec256,testvec256);
   std::cout << "avx512 greater equal 32 bit (v256) " << temp << "\n";
   
   temp=lessequal<avx512< v256< uint64_t > >, 64>(testvec256,testvec256);
   std::cout << "avx512 less equal 64 bit (v256) " << temp << "\n";
   
   temp=lessequal<avx512< v256< uint64_t > >, 32>(testvec256,testvec256);
   std::cout << "avx512 less equal 32 bit (v256) " << temp << "\n";
   
   temp=equality<avx512< v128< uint64_t > >, 64>(testvec128,testvec128);
   std::cout << "avx512 equality 64 bit (v128) " << temp << "\n";
   
   temp=equality<avx512< v128< uint64_t > >, 32>(testvec128,testvec128);
   std::cout << "avx512 equality 32 bit (v128) " << temp << "\n";
   
   temp=lessthan<avx512< v128< uint64_t > >, 64>(testvec128,testvec128);
   std::cout << "avx512 less than 64 bit (v128) " << temp << "\n";
   
   temp=lessthan<avx512< v128< uint64_t > >, 32>(testvec128,testvec128);
   std::cout << "avx512 less than 32 bit (v128) " << temp << "\n";
   
   temp=greaterthan<avx512< v128< uint64_t > >, 64>(testvec128,testvec128);
   std::cout << "avx512 greater than 64 bit (v128) " << temp << "\n";
   
   temp=greaterthan<avx512< v128< uint64_t > >, 32>(testvec128,testvec128);
   std::cout << "avx512 greater than 32 bit (v128) " << temp << "\n";
   
   temp=greaterequal<avx512< v128< uint64_t > >, 64>(testvec128,testvec128);
   std::cout << "avx512 greater equal 64 bit (v128) " << temp << "\n";
   
   temp=greaterequal<avx512< v128< uint64_t > >, 32>(testvec128,testvec128);
   std::cout << "avx512 greater equal 32 bit (v128) " << temp << "\n";
   
   temp=lessequal<avx512< v128< uint64_t > >, 64>(testvec128,testvec128);
   std::cout << "avx512 less equal 64 bit (v128) " << temp << "\n";
   
   temp=lessequal<avx512< v128< uint64_t > >, 32>(testvec128,testvec128);
   std::cout << "avx512 less equal 32 bit (v128) " << temp << "\n";
   
   temp=extract_value<avx512< v512< uint64_t > >, 64>((__m512i)testvec512,0);
   
   std::cout << "avx512 rotate 64 bit (before)" << temp << "\n";
   temp=extract_value<avx512< v512< uint64_t > >, 64>((rotate<avx512< v512< uint64_t > >, 64>(testvec512)),0);
   std::cout << "avx512 rotate 64 bit (after)" << temp << "\n";
   
   temp=extract_value<avx512< v512< uint64_t > >, 64>((set1<avx512< v512< uint64_t > >, 64>(42)),0);
   std::cout << "avx512 set1 64 bit " << temp << "\n";

   temp=extract_value<avx512< v512< uint64_t > >, 64>((set<avx512< v512< uint64_t > >, 64>(49,48,47,46,45,44,43,42)),0);
   std::cout << "avx512 set 64 bit " << temp << "\n";

   temp=extract_value<avx512< v512< uint64_t > >, 32>((set1<avx512< v512< uint64_t > >, 32>(42)),0);
   std::cout << "avx512 set1 32 bit " << temp << "\n";

   temp=extract_value<avx512< v512< uint64_t > >, 32>((set<avx512< v512< uint64_t > >, 32>(57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42)),3);
   std::cout << "avx512 set 32 bit " << temp << "\n";
   
   temp=extract_value<avx512< v512< uint64_t > >, 64>((set_sequence<avx512< v512< uint64_t > >, 64>(0,5)),3);
   std::cout << "avx512 set_sequence 64 bit " << temp << "\n";
   
   temp=extract_value<avx512< v512< uint64_t > >, 32>((set_sequence<avx512< v512< uint64_t > >, 32>(0,5)),3);
   std::cout << "avx512 set_sequence 32 bit " << temp << "\n";
   
   #endif

   return 0;
}