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

namespace storage = morphstore::storage;
namespace format = morphstore::morphing;
namespace operations = morphstore::operators;


void init_data( storage::column< format::uncompr_f > * const perpetualDataColumn ) {
   uint64_t * data = static_cast< uint64_t * >( perpetualDataColumn->data( ) );
   size_t const count = AGGREGATE_SUM_TEST_DATA_COUNT / sizeof( uint64_t );
   for( size_t i = 0; i < count; ++i ) {
      data[ i ] = static_cast< uint64_t >( 1 );
   }
   perpetualDataColumn->set_meta_data( count, AGGREGATE_SUM_TEST_DATA_COUNT );
}


int main( void ) {

   storage::column< format::uncompr_f > * perpetualDataColumn =
      storage::column< format::uncompr_f >::createPerpetualColumn( AGGREGATE_SUM_TEST_DATA_COUNT );
   init_data( perpetualDataColumn );

   std::cout << "Should be "<< AGGREGATE_SUM_TEST_DATA_COUNT / sizeof( uint64_t ) << ". is: " << operations::aggregate_sum( perpetualDataColumn ) << "\n";

   return 0;
}