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
 * @file vbp_routines.h
 * @brief Routines for using the vertical bit-packed layout.
 * @todo Efficient implementations (for now, it must merely work).
 * @todo Somehow include the name of the layout into the way to access these
 *       routines (namespace, struct, name prefix, ...), because we will have
 *       other layouts in the future.
 */

#ifndef MORPHSTORE_CORE_MORPHING_VBP_ROUTINES_H
#define MORPHSTORE_CORE_MORPHING_VBP_ROUTINES_H

#include <core/utils/basic_types.h>

#include <cstdint>
#include <immintrin.h>
#include <limits>

namespace morphstore {
    
    // @todo efficient implementation (for now, it must merely work)
    template< unsigned bw >
    inline void pack( const __m128i * & in128, size_t countIn128, __m128i * & out128 ) {
        __m128i tmp = _mm_setzero_si128( );
        unsigned bitpos = 0;
        const __m128i * const endIn128 = in128 + countIn128;
        const size_t countBits = std::numeric_limits< uint64_t >::digits;
        while( in128 < endIn128 ) {
            while( bitpos + bw <= countBits ) { // as long as the next vector still fits
                tmp = _mm_or_si128( tmp, _mm_slli_epi64( _mm_load_si128( in128++ ), bitpos ) );
                bitpos += bw;
            }
            if( bitpos == countBits ) {
                _mm_store_si128( out128++, tmp );
                tmp = _mm_setzero_si128( );
                bitpos = 0;
            }
            else { // bitpos < countBits
                const __m128i tmp2 = _mm_load_si128( in128++ );
                tmp = _mm_or_si128( tmp, _mm_slli_epi64( tmp2, bitpos ) );
                _mm_store_si128( out128++, tmp );
                tmp = _mm_srli_epi64( tmp2, countBits - bitpos );
                bitpos = bitpos + bw - countBits;
            }
        }
    }

    // @todo efficient implementation (for now, it must merely work)
    template< unsigned bw >
    inline void unpack( const __m128i * & in128, __m128i * & out128, size_t countOut128 ) {
        const size_t countBits = std::numeric_limits< uint64_t >::digits;
        const __m128i mask = _mm_set1_epi64x(
            ( bw == countBits )
            ? std::numeric_limits< uint64_t >::max( )
            : ( static_cast< uint64_t>( 1 ) << bw ) - 1
        );

        __m128i tmp;
        unsigned bitpos = countBits;
        const __m128i * const endOut128 = out128 + countOut128;
        while( out128 < endOut128 ) {
            if( bitpos == countBits ) {
                tmp = _mm_load_si128( in128++ );
                bitpos = 0;
            }
            else { // bitpos < countBits
                const __m128i tmp2 = _mm_load_si128( in128++ );
                _mm_store_si128(
                    out128++,
                    _mm_and_si128(
                        mask,
                        _mm_or_si128(
                            _mm_slli_epi64( tmp2, countBits - bitpos ),
                            _mm_srli_epi64( tmp, bitpos )
                        )
                    )
                );
                tmp = tmp2;
                bitpos = bitpos + bw - countBits;
            }
            while( bitpos + bw <= countBits ) {
                _mm_store_si128(
                    out128++,
                    _mm_and_si128(
                        mask,
                        _mm_srli_epi64( tmp, bitpos )
                    )
                );
                bitpos += bw;
            }
        }
    }
    
}

#endif //MORPHSTORE_CORE_MORPHING_VBP_ROUTINES_H