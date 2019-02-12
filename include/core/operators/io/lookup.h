/**
 * @file lookup.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_IO_LOOKUP_H
#define MORPHSTORE_CORE_OPERATORS_IO_LOOKUP_H

#include "../../storage/column.h"
#include "../../morphing/static_vbp.h"
#include "../../utils/math.h"
#include <cassert>

namespace morphstore { namespace operators {

void lookup(
   storage::column<morphing::uncompr_f> const * const p_DataColumn,
   storage::column<morphing::uncompr_f> const * const p_PositionColumn,
   storage::column<morphing::uncompr_f> * const p_ResultColumn
) {
   assert( p_ResultColumn->count_values( ) == p_PositionColumn->count_values( ) );
   //@todo: Thus we are using only sse, there are no gather loads... so it has to be done in a scalar fashion
   //@todo: ONLY 64-Bit elements
   size_t const positionCount = p_PositionColumn->count_values();
   uint64_t const * const data = p_DataColumn->data( );
   uint64_t const * const positions = p_PositionColumn->data( );
   uint64_t * result = p_ResultColumn->data( );
   for( size_t i = 0; i < positionCount; ++i ) {
      *result++ = data[ positions[ i ] ];
   }
}

/**
 * @todo: support different bitwidths for in and out.
 */
template< uint8_t Bw >
void lookup(
   storage::column< morphing::static_vbp_f< Bw > > const * const p_DataColumn,
   storage::column< morphing::uncompr_f > const * const p_PositionColumn,
   storage::column< morphing::uncompr_f > * const p_ResultColumn
) {
   assert( p_ResultColumn->count_values( ) == p_PositionColumn->count_values( ) );
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
   size_t const positionCount = p_PositionColumn->count_values();
   uint64_t const * const data = p_DataColumn->data( );
   uint64_t const * const positions = p_PositionColumn->data( );
   uint64_t * result = p_ResultColumn->data( );
   for( size_t i = 0; i < positionCount; ++i ) {
      posValue = positions[ i ];
      posInData = posValue * Bw;
      *result++ = ( data[ ( ( posInData ) / 64 ) + ( posValue & 1 ) ] >> ( ( posInData / 2 ) & 63 ) ) && mask;
   }
}


}}
#endif //MORPHSTORE_CORE_OPERATORS_IO_LOOKUP_H
