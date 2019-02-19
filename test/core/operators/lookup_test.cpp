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
 * @file lookup_test.cpp
 * @brief Brief description
 * @todo TODOS?
 */

#include <core/memory/mm_glob.h>
#include <core/storage/column.h>
#include <core/operators/io/lookup.h>

#include <iostream>

#define LOOKUP_TEST_DATA_COUNT 100000000

using namespace morphstore;

void init_data( column< uncompr_f > * const perpetualDataColumn ) {
   uint64_t * data = perpetualDataColumn->get_data( );
   size_t const count = LOOKUP_TEST_DATA_COUNT / sizeof( uint64_t );
   for( size_t i = 0; i < count; ++i ) {
      data[ i ] = static_cast< uint64_t >( i );
   }
}


int main( void ) {

   column< uncompr_f > * perpetualDataColumn =
      column< uncompr_f >::create_perpetual_column( LOOKUP_TEST_DATA_COUNT );
   init_data( perpetualDataColumn );

   column< uncompr_f > ephimeralPositionColumn( 10 * sizeof( uint64_t ) );
   ephimeralPositionColumn.set_meta_data( 10, 10 * sizeof( uint64_t ) );

   column< uncompr_f > ephimeralResultColumn( 10 * sizeof( uint64_t ) );
   ephimeralResultColumn.set_meta_data( 10, 10 * sizeof( uint64_t ) );


   uint64_t * positions = ephimeralPositionColumn.get_data( );

   for( size_t i = 0; i < 10; ++i ) {
      positions[ i ] = 100 * i;
   }

   lookup( perpetualDataColumn, &ephimeralPositionColumn, &ephimeralResultColumn );

   uint64_t * positionsnew = ephimeralPositionColumn.get_data( );
   uint64_t * result = ephimeralResultColumn.get_data( );
   for( size_t i = 0; i < 10; ++i ) {
      std::cout << "should be: " << ( unsigned long ) positionsnew[ i ] << ". is: " << ( unsigned long ) result[ i ] << "\n";
   }
   return 0;
}
