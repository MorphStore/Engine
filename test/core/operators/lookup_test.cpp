/**
 * @file lookup_test.cpp
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#include "../../../include/core/memory/mm_glob.h"
#include "../../../include/core/storage/column.h"
#include "../../../include/core/operators/io/lookup.h"

#include <iostream>

#define LOOKUP_TEST_DATA_COUNT 100000000

namespace storage = morphstore::storage;
namespace format = morphstore::morphing;
namespace operations = morphstore::operators;


void init_data( storage::column< format::uncompr_f > * const perpetualDataColumn ) {
   uint64_t * data = perpetualDataColumn->data( );
   size_t const count = LOOKUP_TEST_DATA_COUNT / sizeof( uint64_t );
   for( size_t i = 0; i < count; ++i ) {
      data[ i ] = static_cast< uint64_t >( i );
   }
}


int main( void ) {

   storage::column< format::uncompr_f > * perpetualDataColumn =
      storage::column< format::uncompr_f >::createPerpetualColumn( LOOKUP_TEST_DATA_COUNT );
   init_data( perpetualDataColumn );

   storage::column< format::uncompr_f > ephimeralPositionColumn( 10 * sizeof( uint64_t ) );
   ephimeralPositionColumn.set_meta_data( 10, 10 * sizeof( uint64_t ) );

   storage::column< format::uncompr_f > ephimeralResultColumn( 10 * sizeof( uint64_t ) );
   ephimeralResultColumn.set_meta_data( 10, 10 * sizeof( uint64_t ) );


   uint64_t * positions = ephimeralPositionColumn.data( );

   for( size_t i = 0; i < 10; ++i ) {
      positions[ i ] = 100 * i;
   }

   operations::lookup( perpetualDataColumn, &ephimeralPositionColumn, &ephimeralResultColumn );

   uint64_t * positionsnew = ephimeralPositionColumn.data( );
   uint64_t * result = ephimeralResultColumn.data( );
   for( size_t i = 0; i < 10; ++i ) {
      std::cout << "should be: " << ( unsigned long ) positionsnew[ i ] << ". is: " << ( unsigned long ) result[ i ] << "\n";
   }
   return 0;
}