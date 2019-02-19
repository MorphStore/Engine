/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/

/**
 * @file summation_test.cpp
 * @brief Brief description
 * @todo TODOS?
 */

#include <core/memory/mm_glob.h>
#include <core/storage/column.h>
#include <core/operators/aggregation/summation.h>

#include <iostream>

#define AGGREGATE_SUM_TEST_DATA_COUNT 100000000

using namespace morphstore;


void init_data( column< uncompr_f > * const perpetualDataColumn ) {
   uint64_t * data = perpetualDataColumn->get_data( );
   size_t const count = AGGREGATE_SUM_TEST_DATA_COUNT / sizeof( uint64_t );
   for( size_t i = 0; i < count; ++i ) {
      data[ i ] = static_cast< uint64_t >( 1 );
   }
   perpetualDataColumn->set_meta_data( count, AGGREGATE_SUM_TEST_DATA_COUNT );
}


int main( void ) {

   column< uncompr_f > * perpetualDataColumn =
      column<uncompr_f>::create_global_column(AGGREGATE_SUM_TEST_DATA_COUNT);
   init_data( perpetualDataColumn );

   std::cout << "Should be "<< AGGREGATE_SUM_TEST_DATA_COUNT / sizeof( uint64_t ) << ". is: " << aggregate_sum( perpetualDataColumn ) << "\n";

   return 0;
}
