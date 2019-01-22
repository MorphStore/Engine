/**
 * @file memory_test.cpp
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#include "../../../include/core/memory/mm_glob.h"

#include <cstdio>
int main( void ) {
   for( int i = 0; i < 53; ++i ) {
      void * result = malloc( 1024 );
      fprintf(stderr, "Allocated: %p\n", result );
   }
   return 0;
}