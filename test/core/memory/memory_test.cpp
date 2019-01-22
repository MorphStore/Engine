/**
 * @file memory_test.cpp
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#define DEBUG_MALLOC
#include "../../../include/core/memory/mm_glob.h"

#include <cstdio>
#include <vector>
#include <iostream>

int main( void ) {
/*   for( int i = 0; i < 53; ++i ) {
      void * result = malloc( 1024 );
      fprintf(stderr, "Main: Allocated: %p\n", result );
   }*/
   std::vector< int > test;
   for( int i = 0; i < 100; ++i ) {
      test.push_back( i );
   }
   for( int i = 0; i < 100; ++i ) {
      std::cout << test[ i ] << "\n";
   }
   return 0;
}