/**
 * @file main.cpp
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#include <vector/general_vector.h>
#include <vector/simd/sse/primitives/io_sse.h>

int main( void ) {
   using namespace vector;
   uint32_t * const data = (uint32_t*)_mm_malloc( 128, 16 );
   typename sse<v128<uint32_t>>::vector_t a =
      load< sse< v128< uint32_t > >, iov::ALIGNED, 128 >::apply( data );
   typename sse< v128< double > >::vector_t b =
      load< sse< v128< double > >, iov::UNALIGNED, 128 >::apply( reinterpret_cast< double * >( data ) );
   _mm_free( data );
   return 0;
}