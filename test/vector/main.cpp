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
#include <iostream>
#include <immintrin.h>



int main( void ) {
    

   using namespace vector;
 
   uint32_t * const data = (uint32_t*)_mm_malloc( 128, 16 );
   data[0]=4;

   int temp=_mm_extract_epi32((load<sse< v128< uint32_t > >, iov::ALIGNED, 128>(data)),0);
   std::cout << "hi " << temp << "\n";
   //   uint32_t * const data = (uint32_t*)_mm_malloc( 128, 16 );
//   typename sse<v128<uint32_t>>::vector_t a =
//      foo< sse< v128< uint32_t > >, iov::ALIGNED, 128 >( data );
//      io< sse< v128< uint32_t > >, iov::ALIGNED, 128 >::load( data );
//   typename sse< v128< double > >::vector_t b =
//      io< sse< v128< double > >, iov::UNALIGNED, 128 >::load( reinterpret_cast< double * >( data ) );
   _mm_free( data );

   return 0;
}