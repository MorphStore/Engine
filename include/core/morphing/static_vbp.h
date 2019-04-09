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
 * @file static_vbp.h
 * @brief A compressed format using the vertical bit-packed layout with a fixed
 *        bit width for an entire column and the corresponding morph operators
 *        for compression and decompression.
 */

#ifndef MORPHSTORE_CORE_MORPHING_STATIC_VBP_H
#define MORPHSTORE_CORE_MORPHING_STATIC_VBP_H

#include <core/morphing/format.h>
#include <core/morphing/vbp_routines.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>

#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace morphstore {
    
    // The vertical bit packed format with a static bit width.
    template< unsigned bw >
    struct static_vbp_f : public format {
        static_assert(
           (1 <= bw) && (bw <= std::numeric_limits< uint64_t >::digits),
           "static_vbp: template parameter bw must satisfy 1 <= bw <= 64"
        );
    };
    
    template< unsigned bw >
    void morph(
        const column< uncompr_f > * in,
        column< static_vbp_f< bw > > * out
    ) {
        // TODO support arbitrary numbers of data elements
        if( in->get_count_values( ) % 128 )
            throw std::runtime_error(
                "morph uncompr_f -> static_vbp_f: the number of data elements "
                "must be a multiple of 128"
            );
        
        const __m128i * in128 = in->get_data( );
        __m128i * out128 = out->get_data( );
        const __m128i * const initOut128 = out128;
        
        pack< bw >(
            in128,
            in->get_size_used_byte( ) / sizeof( __m128i ),
            out128
        );
        
        out->set_count_values( in->get_count_values( ) );
        out->set_size_used_byte( ( out128 - initOut128 ) * sizeof( __m128i ) );
    }
    
    template< unsigned bw >
    void morph(
        const column< static_vbp_f< bw > > * in,
        column< uncompr_f > * out
    ) {
        // TODO support arbitrary numbers of data elements
        if( in->get_count_values( ) % 128 )
            throw std::runtime_error(
                "morph uncompr_f -> static_vbp_f: the number of data elements "
                "must be a multiple of 128"
            );
        
        const __m128i * in128 = in->get_data( );
        __m128i * out128 = out->get_data( );
        const __m128i * const initOut128 = out128;
        
        unpack< bw >(
            in128,
            out128,
            in->get_count_values( ) * sizeof( uint64_t ) / sizeof( __m128i )
        );
        
        out->set_count_values( in->get_count_values( ) );
        out->set_size_used_byte( ( out128 - initOut128 ) * sizeof( __m128i ) );
    }
    
}
#endif //MORPHSTORE_CORE_MORPHING_STATIC_VBP_H
