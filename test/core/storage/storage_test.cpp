/**
 * @file storage_test.cpp
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#include "../../../include/core/memory/mm_glob.h"
#include "../../../include/core/storage/storage_container_uncompressed.h"


#include <cstdio>
#include <vector>
#include <iostream>

int main( void ) {

   using namespace morphstore;

   size_t size = 100_MB;
   size_t sizeb = 200_MB;
   size_t sizec = 300_MB;
   size_t count = size / sizeof( uint32_t );
   uint32_t * dataPtr = ( uint32_t * ) memory::general_memory_manager::get_instance().allocate_persist( size );

   for( size_t i = 0; i < count; ++i ) {
      dataPtr[ i ] = ( uint32_t ) i;
   }
   storage::storage_container_uncompressed< uint32_t > a( storage::storage_persistence_type::PERPETUAL, count, sizeb );

   storage::storage_container_uncompressed< uint32_t > b{ storage::storage_persistence_type::EPHEMERAL, count, sizec };


   return 0;
}