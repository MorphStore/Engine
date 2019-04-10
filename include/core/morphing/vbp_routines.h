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

#if 0
        // This variant uses a store instruction at two points.
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
#else
        // This variant uses a store instruction at only one point.
        __m128i nextOut = _mm_setzero_si128( );
        unsigned bitpos = countBits + bw;
        const __m128i * const endOut128 = out128 + countOut128;
        while( out128 < endOut128 ) {
            __m128i tmp;
            if( bitpos == countBits + bw ) {
                tmp = _mm_load_si128( in128++ );
                nextOut = _mm_and_si128( mask, tmp );
                bitpos = bw;
            }
            else { // bitpos > countBits && bitpos < countBits + bw
                tmp = _mm_load_si128( in128++ );
                nextOut = _mm_and_si128(
                    mask,
                    _mm_or_si128(
                        _mm_slli_epi64( tmp, countBits - bitpos + bw ),
                        nextOut
                    )
                );
                bitpos = bitpos - countBits;
            }
            while( bitpos <= countBits ) {
                _mm_store_si128( out128++, nextOut );
                nextOut = _mm_and_si128(
                    mask,
                    _mm_srli_epi64( tmp, bitpos )
                );
                bitpos += bw;
            }
        }
#endif
    }
    
    inline void pack_switch(
            unsigned bitwidth,
            const __m128i * & in128,
            size_t inCount128,
            __m128i * & out128
    ) {
        switch(bitwidth) {
            // Generated with Python:
            // for bw in range(1, 64+1):
            //   print("case {: >2}: pack<{: >2}>(in128, inCount128, out128); break;".format(bw, bw))
            case  1: pack< 1>(in128, inCount128, out128); break;
            case  2: pack< 2>(in128, inCount128, out128); break;
            case  3: pack< 3>(in128, inCount128, out128); break;
            case  4: pack< 4>(in128, inCount128, out128); break;
            case  5: pack< 5>(in128, inCount128, out128); break;
            case  6: pack< 6>(in128, inCount128, out128); break;
            case  7: pack< 7>(in128, inCount128, out128); break;
            case  8: pack< 8>(in128, inCount128, out128); break;
            case  9: pack< 9>(in128, inCount128, out128); break;
            case 10: pack<10>(in128, inCount128, out128); break;
            case 11: pack<11>(in128, inCount128, out128); break;
            case 12: pack<12>(in128, inCount128, out128); break;
            case 13: pack<13>(in128, inCount128, out128); break;
            case 14: pack<14>(in128, inCount128, out128); break;
            case 15: pack<15>(in128, inCount128, out128); break;
            case 16: pack<16>(in128, inCount128, out128); break;
            case 17: pack<17>(in128, inCount128, out128); break;
            case 18: pack<18>(in128, inCount128, out128); break;
            case 19: pack<19>(in128, inCount128, out128); break;
            case 20: pack<20>(in128, inCount128, out128); break;
            case 21: pack<21>(in128, inCount128, out128); break;
            case 22: pack<22>(in128, inCount128, out128); break;
            case 23: pack<23>(in128, inCount128, out128); break;
            case 24: pack<24>(in128, inCount128, out128); break;
            case 25: pack<25>(in128, inCount128, out128); break;
            case 26: pack<26>(in128, inCount128, out128); break;
            case 27: pack<27>(in128, inCount128, out128); break;
            case 28: pack<28>(in128, inCount128, out128); break;
            case 29: pack<29>(in128, inCount128, out128); break;
            case 30: pack<30>(in128, inCount128, out128); break;
            case 31: pack<31>(in128, inCount128, out128); break;
            case 32: pack<32>(in128, inCount128, out128); break;
            case 33: pack<33>(in128, inCount128, out128); break;
            case 34: pack<34>(in128, inCount128, out128); break;
            case 35: pack<35>(in128, inCount128, out128); break;
            case 36: pack<36>(in128, inCount128, out128); break;
            case 37: pack<37>(in128, inCount128, out128); break;
            case 38: pack<38>(in128, inCount128, out128); break;
            case 39: pack<39>(in128, inCount128, out128); break;
            case 40: pack<40>(in128, inCount128, out128); break;
            case 41: pack<41>(in128, inCount128, out128); break;
            case 42: pack<42>(in128, inCount128, out128); break;
            case 43: pack<43>(in128, inCount128, out128); break;
            case 44: pack<44>(in128, inCount128, out128); break;
            case 45: pack<45>(in128, inCount128, out128); break;
            case 46: pack<46>(in128, inCount128, out128); break;
            case 47: pack<47>(in128, inCount128, out128); break;
            case 48: pack<48>(in128, inCount128, out128); break;
            case 49: pack<49>(in128, inCount128, out128); break;
            case 50: pack<50>(in128, inCount128, out128); break;
            case 51: pack<51>(in128, inCount128, out128); break;
            case 52: pack<52>(in128, inCount128, out128); break;
            case 53: pack<53>(in128, inCount128, out128); break;
            case 54: pack<54>(in128, inCount128, out128); break;
            case 55: pack<55>(in128, inCount128, out128); break;
            case 56: pack<56>(in128, inCount128, out128); break;
            case 57: pack<57>(in128, inCount128, out128); break;
            case 58: pack<58>(in128, inCount128, out128); break;
            case 59: pack<59>(in128, inCount128, out128); break;
            case 60: pack<60>(in128, inCount128, out128); break;
            case 61: pack<61>(in128, inCount128, out128); break;
            case 62: pack<62>(in128, inCount128, out128); break;
            case 63: pack<63>(in128, inCount128, out128); break;
            case 64: pack<64>(in128, inCount128, out128); break;
        }
    }
    
    inline void unpack_switch(
            unsigned bitwidth,
            const __m128i * & in128,
            __m128i * & out128,
            size_t outCount128
    ) {
        switch(bitwidth) {
            // Generated with Python:
            // for bw in range(1, 64+1):
            //   print("case {: >2}: unpack<{: >2}>(in128, out128, outCount128); break;".format(bw, bw))
            case  1: unpack< 1>(in128, out128, outCount128); break;
            case  2: unpack< 2>(in128, out128, outCount128); break;
            case  3: unpack< 3>(in128, out128, outCount128); break;
            case  4: unpack< 4>(in128, out128, outCount128); break;
            case  5: unpack< 5>(in128, out128, outCount128); break;
            case  6: unpack< 6>(in128, out128, outCount128); break;
            case  7: unpack< 7>(in128, out128, outCount128); break;
            case  8: unpack< 8>(in128, out128, outCount128); break;
            case  9: unpack< 9>(in128, out128, outCount128); break;
            case 10: unpack<10>(in128, out128, outCount128); break;
            case 11: unpack<11>(in128, out128, outCount128); break;
            case 12: unpack<12>(in128, out128, outCount128); break;
            case 13: unpack<13>(in128, out128, outCount128); break;
            case 14: unpack<14>(in128, out128, outCount128); break;
            case 15: unpack<15>(in128, out128, outCount128); break;
            case 16: unpack<16>(in128, out128, outCount128); break;
            case 17: unpack<17>(in128, out128, outCount128); break;
            case 18: unpack<18>(in128, out128, outCount128); break;
            case 19: unpack<19>(in128, out128, outCount128); break;
            case 20: unpack<20>(in128, out128, outCount128); break;
            case 21: unpack<21>(in128, out128, outCount128); break;
            case 22: unpack<22>(in128, out128, outCount128); break;
            case 23: unpack<23>(in128, out128, outCount128); break;
            case 24: unpack<24>(in128, out128, outCount128); break;
            case 25: unpack<25>(in128, out128, outCount128); break;
            case 26: unpack<26>(in128, out128, outCount128); break;
            case 27: unpack<27>(in128, out128, outCount128); break;
            case 28: unpack<28>(in128, out128, outCount128); break;
            case 29: unpack<29>(in128, out128, outCount128); break;
            case 30: unpack<30>(in128, out128, outCount128); break;
            case 31: unpack<31>(in128, out128, outCount128); break;
            case 32: unpack<32>(in128, out128, outCount128); break;
            case 33: unpack<33>(in128, out128, outCount128); break;
            case 34: unpack<34>(in128, out128, outCount128); break;
            case 35: unpack<35>(in128, out128, outCount128); break;
            case 36: unpack<36>(in128, out128, outCount128); break;
            case 37: unpack<37>(in128, out128, outCount128); break;
            case 38: unpack<38>(in128, out128, outCount128); break;
            case 39: unpack<39>(in128, out128, outCount128); break;
            case 40: unpack<40>(in128, out128, outCount128); break;
            case 41: unpack<41>(in128, out128, outCount128); break;
            case 42: unpack<42>(in128, out128, outCount128); break;
            case 43: unpack<43>(in128, out128, outCount128); break;
            case 44: unpack<44>(in128, out128, outCount128); break;
            case 45: unpack<45>(in128, out128, outCount128); break;
            case 46: unpack<46>(in128, out128, outCount128); break;
            case 47: unpack<47>(in128, out128, outCount128); break;
            case 48: unpack<48>(in128, out128, outCount128); break;
            case 49: unpack<49>(in128, out128, outCount128); break;
            case 50: unpack<50>(in128, out128, outCount128); break;
            case 51: unpack<51>(in128, out128, outCount128); break;
            case 52: unpack<52>(in128, out128, outCount128); break;
            case 53: unpack<53>(in128, out128, outCount128); break;
            case 54: unpack<54>(in128, out128, outCount128); break;
            case 55: unpack<55>(in128, out128, outCount128); break;
            case 56: unpack<56>(in128, out128, outCount128); break;
            case 57: unpack<57>(in128, out128, outCount128); break;
            case 58: unpack<58>(in128, out128, outCount128); break;
            case 59: unpack<59>(in128, out128, outCount128); break;
            case 60: unpack<60>(in128, out128, outCount128); break;
            case 61: unpack<61>(in128, out128, outCount128); break;
            case 62: unpack<62>(in128, out128, outCount128); break;
            case 63: unpack<63>(in128, out128, outCount128); break;
            case 64: unpack<64>(in128, out128, outCount128); break;
        }
    }
    
}

#endif //MORPHSTORE_CORE_MORPHING_VBP_ROUTINES_H