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
 * @file memory_test.cpp
 * @brief Brief description
 * @todo TODOS?
 */


#include <core/memory/mm_glob.h>


#include <cstdio>
#include <vector>
#include <iostream>

int main( void ) {
/*   for( int i = 0; i < 53; ++i ) {
      void * result = malloc( 1024 );
      fprintf(stderr, "Main: Allocated: %p\n", result );
   }*/


   std::vector< int > test;
   int * a = (int*)malloc( 400 );
   for( int i = 0; i < 20; ++i ) {
      test.push_back( i );
   }
   for( int i = 0; i < 100; ++i ) {
      std::cout << test[ i ] << ";";
      a[ i ] = i;
   }
   std::cout << "\n";
   return 0;
}
