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
   size_t count = size / sizeof( uint32_t );
   uint32_t * dataPtr = ( uint32_t * ) memory::general_memory_manager::get_instance().allocate_persist( size );

   for( size_t i = 0; i < count; ++i ) {
      dataPtr[ i ] = ( uint32_t ) i;
   }
   storage::storage_container_uncompressed< uint32_t > a{
      storage::storage_container_meta_data< uint32_t >{
         count, size
      },
      dataPtr
   };


   return 0;
}