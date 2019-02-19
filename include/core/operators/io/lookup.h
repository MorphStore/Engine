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
 * @file lookup.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_IO_LOOKUP_H
#define MORPHSTORE_CORE_OPERATORS_IO_LOOKUP_H

#include <core/storage/column.h>
#include <core/morphing/static_vbp.h>
#include <core/utils/math.h>
#include <cassert>

namespace morphstore {

void lookup(
   column<uncompr_f> const * const p_DataColumn,
   column<uncompr_f> const * const p_PositionColumn,
   column<uncompr_f> * const p_ResultColumn
) {
   assert( p_ResultColumn->get_count_values( ) == p_PositionColumn->get_count_values( ) );
   //@todo: Thus we are using only sse, there are no gather loads... so it has to be done in a scalar fashion
   //@todo: ONLY 64-Bit elements
   size_t const positionCount = p_PositionColumn->get_count_values();
   uint64_t const * const data = p_DataColumn->get_data( );
   uint64_t const * const positions = p_PositionColumn->get_data( );
   uint64_t * result = p_ResultColumn->get_data( );
   for( size_t i = 0; i < positionCount; ++i ) {
      *result++ = data[ positions[ i ] ];
   }
}

/**
 * @todo: support different bitwidths for in and out.
 */
template< uint8_t Bw >
void lookup(
   column< static_vbp_f< Bw > > const * const p_DataColumn,
   column< uncompr_f > const * const p_PositionColumn,
   column< uncompr_f > * const p_ResultColumn
) {
   assert( p_ResultColumn->get_count_values( ) == p_PositionColumn->get_count_values( ) );
   /**
    * pos:      current position from p_PositionColumn
    * Bw:       bitwidth
    * vecpos:   n-th vector register (128-bit wide)
    * wordpos:  n-th word (64-bit wide) in vector register (128-bit wide)
    * bitpos:   position within word
    *
    * vecpos    = ( pos * bw / 128 ) * 2    // we need to multiply by 2 since 2 64-bit words fit into one vector
    * wordpos   = pos & 1                   // even values are located in word pos 0, odd values in word pos 1
    * bitpos    = ( pos / 2 * bw ) & 63
    *
    */

   uint64_t const mask = to_the_power_of_two( Bw ) - 1;
   uint64_t posValue = 0;
   uint64_t posInData = 0;

   //@todo: Thus we are using only sse, there are no gather loads... so it has to be done in a scalar fashion
   //@todo: ONLY 64-Bit elements
   size_t const positionCount = p_PositionColumn->get_count_values();
   uint64_t const * const data = p_DataColumn->get_data( );
   uint64_t const * const positions = p_PositionColumn->get_data( );
   uint64_t * result = p_ResultColumn->get_data( );
   for( size_t i = 0; i < positionCount; ++i ) {
      posValue = positions[ i ];
      posInData = posValue * Bw;
      *result++ = ( data[ ( ( posInData ) / 64 ) + ( posValue & 1 ) ] >> ( ( posInData / 2 ) & 63 ) ) && mask;
   }
}


}
#endif //MORPHSTORE_CORE_OPERATORS_IO_LOOKUP_H
