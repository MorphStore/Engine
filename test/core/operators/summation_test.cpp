/**
 * @file summation_test.cpp
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#include "../../../include/core/memory/mm_glob.h"
#include "../../../include/core/storage/column.h"
#include "../../../include/core/operators/aggregation/summation.h"

#include <iostream>

#define AGGREGATE_SUM_TEST_DATA_COUNT 100000000

using namespace morphstore;


void init_data( column< uncompr_f > * const perpetualDataColumn ) {
   uint64_t * data = perpetualDataColumn->data( );
   size_t const count = AGGREGATE_SUM_TEST_DATA_COUNT / sizeof( uint64_t );
   for( size_t i = 0; i < count; ++i ) {
      data[ i ] = static_cast< uint64_t >( 1 );
   }
   perpetualDataColumn->set_meta_data( count, AGGREGATE_SUM_TEST_DATA_COUNT );
}


int main( void ) {

   column< uncompr_f > * perpetualDataColumn =
      column< uncompr_f >::createPerpetualColumn( AGGREGATE_SUM_TEST_DATA_COUNT );
   init_data( perpetualDataColumn );

   std::cout << "Should be "<< AGGREGATE_SUM_TEST_DATA_COUNT / sizeof( uint64_t ) << ". is: " << aggregate_sum( perpetualDataColumn ) << "\n";

   return 0;
}