/**
 * @file storage_test.cpp
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#include "../../../include/core/memory/mm_glob.h"
#include "../../../include/core/storage/column.h"


#include <cstdio>
#include <vector>
#include <iostream>

int main( void ) {

   using namespace morphstore;

   storage::column< uint32_t > a{ storage::storage_persistence_type::PERPETUAL, 100 };


   return 0;
}