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
 * @todo It would be great if we could derive the vector datatype (e.g.
 *       __m128i) from the processing style. Then we would not require uint8_t*
 *       parameters and it would also be easier for the callers.
 * @todo Documentation.
 */

#ifndef MORPHSTORE_CORE_MORPHING_VBP_ROUTINES_H
#define MORPHSTORE_CORE_MORPHING_VBP_ROUTINES_H

#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <core/utils/processing_style.h>

#include <cstdint>
#include <immintrin.h>
#include <limits>

namespace morphstore {
    
    // ************************************************************************
    // Packing routines
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Interfaces
    // ------------------------------------------------------------------------
    
    // Struct for partial template specialization.
    template<
            processing_style_t t_ps,
            unsigned t_bw,
            unsigned t_step
    >
    struct pack_t {
        static MSV_CXX_ATTRIBUTE_FORCE_INLINE void apply(
                const uint8_t * & in8,
                size_t countIn64,
                uint8_t * & out8
        ) = delete;
    };
    
    // Convenience function.
    template<
            processing_style_t t_ps,
            unsigned t_bw,
            unsigned t_step
    >
    MSV_CXX_ATTRIBUTE_FORCE_INLINE void pack(
            const uint8_t * & in8,
            size_t countIn64,
            uint8_t * & out8
    ) {
        pack_t<t_ps, t_bw, t_step>::apply(in8, countIn64, out8);
    }
    
    
    // ------------------------------------------------------------------------
    // Template specializations.
    // ------------------------------------------------------------------------
    
    template<unsigned t_bw>
    struct pack_t<
            processing_style_t::vec128,
            t_bw,
            sizeof(__m128i) / sizeof(uint64_t)
    > {
        static MSV_CXX_ATTRIBUTE_FORCE_INLINE void apply(
                const uint8_t * & in8,
                size_t countIn64,
                uint8_t * & out8
        ) {
            const __m128i * in128 = reinterpret_cast<const __m128i *>(in8);
            const size_t countIn128 = convert_size<uint64_t, __m128i>(countIn64);
            __m128i * out128 = reinterpret_cast<__m128i *>(out8);

            __m128i tmpA = _mm_setzero_si128();
            unsigned bitpos = 0;
            const __m128i * const endIn128 = in128 + countIn128;
            const size_t countBits = std::numeric_limits<uint64_t>::digits;
            while(in128 < endIn128) {
                while(bitpos + t_bw <= countBits) { // as long as the next vector still fits
                    tmpA = _mm_or_si128(
                            tmpA,
                            _mm_slli_epi64(
                                    _mm_load_si128(in128++),
                                    bitpos
                            )
                    );
                    bitpos += t_bw;
                }
                if(bitpos == countBits) {
                    _mm_store_si128(out128++, tmpA);
                    tmpA = _mm_setzero_si128();
                    bitpos = 0;
                }
                else { // bitpos < countBits
                    const __m128i tmpB = _mm_load_si128(in128++);
                    tmpA = _mm_or_si128(tmpA, _mm_slli_epi64(tmpB, bitpos));
                    _mm_store_si128(out128++, tmpA);
                    tmpA = _mm_srli_epi64(tmpB, countBits - bitpos);
                    bitpos = bitpos + t_bw - countBits;
                }
            }

            in8 = reinterpret_cast<const uint8_t *>(in128);
            out8 = reinterpret_cast<uint8_t *>(out128);
        };
    };
    
#if 0
    // Tailored to a step width of 2 (as for the 128-bit variant).
    template<unsigned t_bw>
    struct pack_t<
            processing_style_t::scalar,
            t_bw,
            sizeof(__m128i) / sizeof(uint64_t)
    > {
        static MSV_CXX_ATTRIBUTE_FORCE_INLINE void apply(
                const uint8_t * & in8,
                size_t countIn64,
                uint8_t * & out8
        ) {
            const uint64_t * in64 = reinterpret_cast<const uint64_t *>(in8);
            uint64_t * out64 = reinterpret_cast<uint64_t *>(out8);

            uint64_t tmpA0 = 0;
            uint64_t tmpA1 = 0;
            unsigned bitpos = 0;
            const uint64_t * const endIn64 = in64 + countIn64;
            const size_t countBits = std::numeric_limits<uint64_t>::digits;
            while(in64 < endIn64) {
                while(bitpos + t_bw <= countBits) { // as long as the next vector still fits
                    tmpA0 |= ((*in64++) << bitpos);
                    tmpA1 |= ((*in64++) << bitpos);
                    bitpos += t_bw;
                }
                if(bitpos == countBits) {
                    *out64++ = tmpA0;
                    *out64++ = tmpA1;
                    tmpA0 = 0;
                    tmpA1 = 0;
                    bitpos = 0;
                }
                else { // bitpos < countBits
                    const uint64_t tmpB0 = *in64++;
                    const uint64_t tmpB1 = *in64++;
                    tmpA0 |= (tmpB0 << bitpos);
                    tmpA1 |= (tmpB1 << bitpos);
                    *out64++ = tmpA0;
                    *out64++ = tmpA1;
                    tmpA0 = (tmpB0 >> (countBits - bitpos));
                    tmpA1 = (tmpB1 >> (countBits - bitpos));
                    bitpos = bitpos + t_bw - countBits;
                }
            }

            in8 = reinterpret_cast<const uint8_t *>(in64);
            out8 = reinterpret_cast<uint8_t *>(out64);
        };
    };
#else
    // Generic w.r.t. the step width. Hopefully the compiler unrolls the loops.
    template<unsigned t_bw, unsigned t_step>
    struct pack_t<
            processing_style_t::scalar,
            t_bw,
            t_step
    > {
        static MSV_CXX_ATTRIBUTE_FORCE_INLINE void apply(
                const uint8_t * & in8,
                size_t countIn64,
                uint8_t * & out8
        ) {
            const uint64_t * in64 = reinterpret_cast<const uint64_t *>(in8);
            uint64_t * out64 = reinterpret_cast<uint64_t *>(out8);

            uint64_t tmpA[t_step];
            for(unsigned i = 0; i < t_step; i++)
                tmpA[i] = 0;
            unsigned bitpos = 0;
            const uint64_t * const endIn64 = in64 + countIn64;
            const size_t countBits = std::numeric_limits<uint64_t>::digits;
            while(in64 < endIn64) {
                while(bitpos + t_bw <= countBits) { // as long as the next vector still fits
                    for(unsigned i = 0; i < t_step; i++)
                        tmpA[i] |= ((*in64++) << bitpos);
                    bitpos += t_bw;
                }
                if(bitpos == countBits) {
                    for(unsigned i = 0; i < t_step; i++) {
                        *out64++ = tmpA[i];
                        tmpA[i] = 0;
                    }
                    bitpos = 0;
                }
                else { // bitpos < countBits
                    for(unsigned i = 0; i < t_step; i++) {
                        const uint64_t tmpB = *in64++;
                        tmpA[i] |= (tmpB << bitpos);
                        *out64++ = tmpA[i];
                        tmpA[i] = (tmpB >> (countBits - bitpos));
                    }
                    bitpos = bitpos + t_bw - countBits;
                }
            }

            in8 = reinterpret_cast<const uint8_t *>(in64);
            out8 = reinterpret_cast<uint8_t *>(out64);
        };
    };
#endif

    
    // ------------------------------------------------------------------------
    // Selection of the right routine at run-time.
    // ------------------------------------------------------------------------
    
    template<
            processing_style_t t_ps,
            unsigned t_step
    >
    MSV_CXX_ATTRIBUTE_FORCE_INLINE void pack_switch(
            unsigned bitwidth,
            const uint8_t * & in8,
            size_t inCount64,
            uint8_t * & out8
    ) {
        switch(bitwidth) {
            // Generated with Python:
            // for bw in range(1, 64+1):
            //   print("case {: >2}: pack<t_ps, {: >2}, t_step>(in8, inCount64, out8); break;".format(bw, bw))
            case  1: pack<t_ps,  1, t_step>(in8, inCount64, out8); break;
            case  2: pack<t_ps,  2, t_step>(in8, inCount64, out8); break;
            case  3: pack<t_ps,  3, t_step>(in8, inCount64, out8); break;
            case  4: pack<t_ps,  4, t_step>(in8, inCount64, out8); break;
            case  5: pack<t_ps,  5, t_step>(in8, inCount64, out8); break;
            case  6: pack<t_ps,  6, t_step>(in8, inCount64, out8); break;
            case  7: pack<t_ps,  7, t_step>(in8, inCount64, out8); break;
            case  8: pack<t_ps,  8, t_step>(in8, inCount64, out8); break;
            case  9: pack<t_ps,  9, t_step>(in8, inCount64, out8); break;
            case 10: pack<t_ps, 10, t_step>(in8, inCount64, out8); break;
            case 11: pack<t_ps, 11, t_step>(in8, inCount64, out8); break;
            case 12: pack<t_ps, 12, t_step>(in8, inCount64, out8); break;
            case 13: pack<t_ps, 13, t_step>(in8, inCount64, out8); break;
            case 14: pack<t_ps, 14, t_step>(in8, inCount64, out8); break;
            case 15: pack<t_ps, 15, t_step>(in8, inCount64, out8); break;
            case 16: pack<t_ps, 16, t_step>(in8, inCount64, out8); break;
            case 17: pack<t_ps, 17, t_step>(in8, inCount64, out8); break;
            case 18: pack<t_ps, 18, t_step>(in8, inCount64, out8); break;
            case 19: pack<t_ps, 19, t_step>(in8, inCount64, out8); break;
            case 20: pack<t_ps, 20, t_step>(in8, inCount64, out8); break;
            case 21: pack<t_ps, 21, t_step>(in8, inCount64, out8); break;
            case 22: pack<t_ps, 22, t_step>(in8, inCount64, out8); break;
            case 23: pack<t_ps, 23, t_step>(in8, inCount64, out8); break;
            case 24: pack<t_ps, 24, t_step>(in8, inCount64, out8); break;
            case 25: pack<t_ps, 25, t_step>(in8, inCount64, out8); break;
            case 26: pack<t_ps, 26, t_step>(in8, inCount64, out8); break;
            case 27: pack<t_ps, 27, t_step>(in8, inCount64, out8); break;
            case 28: pack<t_ps, 28, t_step>(in8, inCount64, out8); break;
            case 29: pack<t_ps, 29, t_step>(in8, inCount64, out8); break;
            case 30: pack<t_ps, 30, t_step>(in8, inCount64, out8); break;
            case 31: pack<t_ps, 31, t_step>(in8, inCount64, out8); break;
            case 32: pack<t_ps, 32, t_step>(in8, inCount64, out8); break;
            case 33: pack<t_ps, 33, t_step>(in8, inCount64, out8); break;
            case 34: pack<t_ps, 34, t_step>(in8, inCount64, out8); break;
            case 35: pack<t_ps, 35, t_step>(in8, inCount64, out8); break;
            case 36: pack<t_ps, 36, t_step>(in8, inCount64, out8); break;
            case 37: pack<t_ps, 37, t_step>(in8, inCount64, out8); break;
            case 38: pack<t_ps, 38, t_step>(in8, inCount64, out8); break;
            case 39: pack<t_ps, 39, t_step>(in8, inCount64, out8); break;
            case 40: pack<t_ps, 40, t_step>(in8, inCount64, out8); break;
            case 41: pack<t_ps, 41, t_step>(in8, inCount64, out8); break;
            case 42: pack<t_ps, 42, t_step>(in8, inCount64, out8); break;
            case 43: pack<t_ps, 43, t_step>(in8, inCount64, out8); break;
            case 44: pack<t_ps, 44, t_step>(in8, inCount64, out8); break;
            case 45: pack<t_ps, 45, t_step>(in8, inCount64, out8); break;
            case 46: pack<t_ps, 46, t_step>(in8, inCount64, out8); break;
            case 47: pack<t_ps, 47, t_step>(in8, inCount64, out8); break;
            case 48: pack<t_ps, 48, t_step>(in8, inCount64, out8); break;
            case 49: pack<t_ps, 49, t_step>(in8, inCount64, out8); break;
            case 50: pack<t_ps, 50, t_step>(in8, inCount64, out8); break;
            case 51: pack<t_ps, 51, t_step>(in8, inCount64, out8); break;
            case 52: pack<t_ps, 52, t_step>(in8, inCount64, out8); break;
            case 53: pack<t_ps, 53, t_step>(in8, inCount64, out8); break;
            case 54: pack<t_ps, 54, t_step>(in8, inCount64, out8); break;
            case 55: pack<t_ps, 55, t_step>(in8, inCount64, out8); break;
            case 56: pack<t_ps, 56, t_step>(in8, inCount64, out8); break;
            case 57: pack<t_ps, 57, t_step>(in8, inCount64, out8); break;
            case 58: pack<t_ps, 58, t_step>(in8, inCount64, out8); break;
            case 59: pack<t_ps, 59, t_step>(in8, inCount64, out8); break;
            case 60: pack<t_ps, 60, t_step>(in8, inCount64, out8); break;
            case 61: pack<t_ps, 61, t_step>(in8, inCount64, out8); break;
            case 62: pack<t_ps, 62, t_step>(in8, inCount64, out8); break;
            case 63: pack<t_ps, 63, t_step>(in8, inCount64, out8); break;
            case 64: pack<t_ps, 64, t_step>(in8, inCount64, out8); break;
        }
    }
    
    
    
    // ************************************************************************
    // Unpacking routines
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Interfaces
    // ------------------------------------------------------------------------
    
    // Struct for partial template specialization.
    template<
            processing_style_t t_ps,
            unsigned t_bw,
            unsigned t_step
    >
    struct unpack_t {
        static MSV_CXX_ATTRIBUTE_FORCE_INLINE void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countOut64
        ) = delete;
    };
    
    // Convenience function.
    template<
            processing_style_t t_ps,
            unsigned t_bw,
            unsigned t_step
    >
    MSV_CXX_ATTRIBUTE_FORCE_INLINE void unpack(
            const uint8_t * & in8,
            uint8_t * & out8,
            size_t countOut64
    ) {
        unpack_t<t_ps, t_bw, t_step>::apply(in8, out8, countOut64);
    }
    
    
    // ------------------------------------------------------------------------
    // Template specializations.
    // ------------------------------------------------------------------------
    
    template<unsigned t_bw>
    struct unpack_t<
            processing_style_t::vec128,
            t_bw,
            sizeof(__m128i) / sizeof(uint64_t)
    > {
        static MSV_CXX_ATTRIBUTE_FORCE_INLINE void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countOut64
        ) {
            const __m128i * in128 = reinterpret_cast<const __m128i *>(in8);
            __m128i * out128 = reinterpret_cast<__m128i *>(out8);
            const size_t countOut128 = convert_size<uint64_t, __m128i>(countOut64);

            const size_t countBits = std::numeric_limits<uint64_t>::digits;
            const __m128i mask = _mm_set1_epi64x(bitwidth_max<uint64_t>(t_bw));

#if 0
            // This variant uses a store instruction at two points.
            __m128i tmp;
            unsigned bitpos = countBits;
            const __m128i * const endOut128 = out128 + countOut128;
            while(out128 < endOut128) {
                if(bitpos == countBits) {
                    tmp = _mm_load_si128(in128++);
                    bitpos = 0;
                }
                else { // bitpos < countBits
                    const __m128i tmp2 = _mm_load_si128(in128++);
                    _mm_store_si128(
                        out128++,
                        _mm_and_si128(
                            mask,
                            _mm_or_si128(
                                _mm_slli_epi64(tmp2, countBits - bitpos),
                                _mm_srli_epi64(tmp, bitpos)
                            )
                        )
                    );
                    tmp = tmp2;
                    bitpos = bitpos + t_bw - countBits;
                }
                while(bitpos + t_bw <= countBits) {
                    _mm_store_si128(
                        out128++,
                        _mm_and_si128(
                            mask,
                            _mm_srli_epi64(tmp, bitpos)
                        )
                    );
                    bitpos += t_bw;
                }
            }
#else
            // This variant uses a store instruction at only one point.
            __m128i nextOut = _mm_setzero_si128();
            unsigned bitpos = countBits + t_bw;
            const __m128i * const endOut128 = out128 + countOut128;
            while(out128 < endOut128) {
                __m128i tmp;
                if(bitpos == countBits + t_bw) {
                    tmp = _mm_load_si128(in128++);
                    nextOut = _mm_and_si128(mask, tmp);
                    bitpos = t_bw;
                }
                else { // bitpos > countBits && bitpos < countBits + t_bw
                    tmp = _mm_load_si128(in128++);
                    nextOut = _mm_and_si128(
                        mask,
                        _mm_or_si128(
                            _mm_slli_epi64(tmp, countBits - bitpos + t_bw),
                            nextOut
                        )
                    );
                    bitpos = bitpos - countBits;
                }
                while(bitpos <= countBits) {
                    _mm_store_si128(out128++, nextOut);
                    nextOut = _mm_and_si128(
                        mask,
                        _mm_srli_epi64(tmp, bitpos)
                    );
                    bitpos += t_bw;
                }
            }
#endif

            in8 = reinterpret_cast<const uint8_t *>(in128);
            out8 = reinterpret_cast<uint8_t *>(out128);
        }
    };
    
#if 0
    // Tailored to a step width of 2 (as for the 128-bit variant).
    template<unsigned t_bw>
    struct unpack_t<
            processing_style_t::scalar,
            t_bw,
            sizeof(__m128i) / sizeof(uint64_t)
    > {
        static MSV_CXX_ATTRIBUTE_FORCE_INLINE void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countOut64
        ) {
            const uint64_t * in64 = reinterpret_cast<const uint64_t *>(in8);
            uint64_t * out64 = reinterpret_cast<uint64_t *>(out8);
            
            const size_t countBits = std::numeric_limits<uint64_t>::digits;
            const uint64_t mask = bitwidth_max<uint64_t>(t_bw);

            // This variant uses a store instruction at only one point.
            uint64_t nextOut0 = 0;
            uint64_t nextOut1 = 0;
            unsigned bitpos = countBits + t_bw;
            const uint64_t * const endOut64 = out64 + countOut64;
            while(out64 < endOut64) {
                uint64_t tmp0;
                uint64_t tmp1;
                if(bitpos == countBits + t_bw) {
                    tmp0 = *in64++;
                    tmp1 = *in64++;
                    nextOut0 = mask & tmp0;
                    nextOut1 = mask & tmp1;
                    bitpos = t_bw;
                }
                else { // bitpos > countBits && bitpos < countBits + t_bw
                    tmp0 = *in64++;
                    tmp1 = *in64++;
                    nextOut0 = mask & ((tmp0 << (countBits - bitpos + t_bw)) | nextOut0);
                    nextOut1 = mask & ((tmp1 << (countBits - bitpos + t_bw)) | nextOut1);
                    bitpos = bitpos - countBits;
                }
                while(bitpos <= countBits) {
                    *out64++ = nextOut0;
                    *out64++ = nextOut1;
                    nextOut0 = mask & (tmp0 >> bitpos);
                    nextOut1 = mask & (tmp1 >> bitpos);
                    bitpos += t_bw;
                }
            }
            
            in8 = reinterpret_cast<const uint8_t *>(in64);
            out8 = reinterpret_cast<uint8_t *>(out64);
        }
    };
#else
    // Generic w.r.t. the step width. Hopefully the compiler unrolls the loops.
    template<unsigned t_bw, unsigned t_step>
    struct unpack_t<
            processing_style_t::scalar,
            t_bw,
            t_step
    > {
        static MSV_CXX_ATTRIBUTE_FORCE_INLINE void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countOut64
        ) {
            const uint64_t * in64 = reinterpret_cast<const uint64_t *>(in8);
            uint64_t * out64 = reinterpret_cast<uint64_t *>(out8);
            
            const size_t countBits = std::numeric_limits<uint64_t>::digits;
            const uint64_t mask = bitwidth_max<uint64_t>(t_bw);

            // This variant uses a store instruction at only one point.
            uint64_t nextOut[t_step];
            for(unsigned i = 0; i < t_step; i++)
                nextOut[i] = 0;
            unsigned bitpos = countBits + t_bw;
            const uint64_t * const endOut64 = out64 + countOut64;
            while(out64 < endOut64) {
                uint64_t tmp[t_step];
                if(bitpos == countBits + t_bw) {
                    for(unsigned i = 0; i < t_step; i++) {
                        tmp[i] = *in64++;
                        nextOut[i] = mask & tmp[i];
                    }
                    bitpos = t_bw;
                }
                else { // bitpos > countBits && bitpos < countBits + t_bw
                    for(unsigned i = 0; i < t_step; i++) {
                        tmp[i] = *in64++;
                        nextOut[i] = mask & ((tmp[i] << (countBits - bitpos + t_bw)) | nextOut[i]);
                    }
                    bitpos = bitpos - countBits;
                }
                while(bitpos <= countBits) {
                    for(unsigned i = 0; i < t_step; i++) {
                        *out64++ = nextOut[i];
                        nextOut[i] = mask & (tmp[i] >> bitpos);
                    }
                    bitpos += t_bw;
                }
            }
            
            in8 = reinterpret_cast<const uint8_t *>(in64);
            out8 = reinterpret_cast<uint8_t *>(out64);
        }
    };
#endif
    
    // ------------------------------------------------------------------------
    // Selection of the right routine at run-time.
    // ------------------------------------------------------------------------
    
    template<
            processing_style_t t_ps,
            unsigned t_step
    >
    MSV_CXX_ATTRIBUTE_FORCE_INLINE void unpack_switch(
            unsigned bitwidth,
            const uint8_t * & in8,
            uint8_t * & out8,
            size_t outCount64
    ) {
        switch(bitwidth) {
            // Generated with Python:
            // for bw in range(1, 64+1):
            //   print("case {: >2}: unpack<t_ps, {: >2}, t_step>(in8, out8, outCount64); break;".format(bw, bw))
            case  1: unpack<t_ps,  1, t_step>(in8, out8, outCount64); break;
            case  2: unpack<t_ps,  2, t_step>(in8, out8, outCount64); break;
            case  3: unpack<t_ps,  3, t_step>(in8, out8, outCount64); break;
            case  4: unpack<t_ps,  4, t_step>(in8, out8, outCount64); break;
            case  5: unpack<t_ps,  5, t_step>(in8, out8, outCount64); break;
            case  6: unpack<t_ps,  6, t_step>(in8, out8, outCount64); break;
            case  7: unpack<t_ps,  7, t_step>(in8, out8, outCount64); break;
            case  8: unpack<t_ps,  8, t_step>(in8, out8, outCount64); break;
            case  9: unpack<t_ps,  9, t_step>(in8, out8, outCount64); break;
            case 10: unpack<t_ps, 10, t_step>(in8, out8, outCount64); break;
            case 11: unpack<t_ps, 11, t_step>(in8, out8, outCount64); break;
            case 12: unpack<t_ps, 12, t_step>(in8, out8, outCount64); break;
            case 13: unpack<t_ps, 13, t_step>(in8, out8, outCount64); break;
            case 14: unpack<t_ps, 14, t_step>(in8, out8, outCount64); break;
            case 15: unpack<t_ps, 15, t_step>(in8, out8, outCount64); break;
            case 16: unpack<t_ps, 16, t_step>(in8, out8, outCount64); break;
            case 17: unpack<t_ps, 17, t_step>(in8, out8, outCount64); break;
            case 18: unpack<t_ps, 18, t_step>(in8, out8, outCount64); break;
            case 19: unpack<t_ps, 19, t_step>(in8, out8, outCount64); break;
            case 20: unpack<t_ps, 20, t_step>(in8, out8, outCount64); break;
            case 21: unpack<t_ps, 21, t_step>(in8, out8, outCount64); break;
            case 22: unpack<t_ps, 22, t_step>(in8, out8, outCount64); break;
            case 23: unpack<t_ps, 23, t_step>(in8, out8, outCount64); break;
            case 24: unpack<t_ps, 24, t_step>(in8, out8, outCount64); break;
            case 25: unpack<t_ps, 25, t_step>(in8, out8, outCount64); break;
            case 26: unpack<t_ps, 26, t_step>(in8, out8, outCount64); break;
            case 27: unpack<t_ps, 27, t_step>(in8, out8, outCount64); break;
            case 28: unpack<t_ps, 28, t_step>(in8, out8, outCount64); break;
            case 29: unpack<t_ps, 29, t_step>(in8, out8, outCount64); break;
            case 30: unpack<t_ps, 30, t_step>(in8, out8, outCount64); break;
            case 31: unpack<t_ps, 31, t_step>(in8, out8, outCount64); break;
            case 32: unpack<t_ps, 32, t_step>(in8, out8, outCount64); break;
            case 33: unpack<t_ps, 33, t_step>(in8, out8, outCount64); break;
            case 34: unpack<t_ps, 34, t_step>(in8, out8, outCount64); break;
            case 35: unpack<t_ps, 35, t_step>(in8, out8, outCount64); break;
            case 36: unpack<t_ps, 36, t_step>(in8, out8, outCount64); break;
            case 37: unpack<t_ps, 37, t_step>(in8, out8, outCount64); break;
            case 38: unpack<t_ps, 38, t_step>(in8, out8, outCount64); break;
            case 39: unpack<t_ps, 39, t_step>(in8, out8, outCount64); break;
            case 40: unpack<t_ps, 40, t_step>(in8, out8, outCount64); break;
            case 41: unpack<t_ps, 41, t_step>(in8, out8, outCount64); break;
            case 42: unpack<t_ps, 42, t_step>(in8, out8, outCount64); break;
            case 43: unpack<t_ps, 43, t_step>(in8, out8, outCount64); break;
            case 44: unpack<t_ps, 44, t_step>(in8, out8, outCount64); break;
            case 45: unpack<t_ps, 45, t_step>(in8, out8, outCount64); break;
            case 46: unpack<t_ps, 46, t_step>(in8, out8, outCount64); break;
            case 47: unpack<t_ps, 47, t_step>(in8, out8, outCount64); break;
            case 48: unpack<t_ps, 48, t_step>(in8, out8, outCount64); break;
            case 49: unpack<t_ps, 49, t_step>(in8, out8, outCount64); break;
            case 50: unpack<t_ps, 50, t_step>(in8, out8, outCount64); break;
            case 51: unpack<t_ps, 51, t_step>(in8, out8, outCount64); break;
            case 52: unpack<t_ps, 52, t_step>(in8, out8, outCount64); break;
            case 53: unpack<t_ps, 53, t_step>(in8, out8, outCount64); break;
            case 54: unpack<t_ps, 54, t_step>(in8, out8, outCount64); break;
            case 55: unpack<t_ps, 55, t_step>(in8, out8, outCount64); break;
            case 56: unpack<t_ps, 56, t_step>(in8, out8, outCount64); break;
            case 57: unpack<t_ps, 57, t_step>(in8, out8, outCount64); break;
            case 58: unpack<t_ps, 58, t_step>(in8, out8, outCount64); break;
            case 59: unpack<t_ps, 59, t_step>(in8, out8, outCount64); break;
            case 60: unpack<t_ps, 60, t_step>(in8, out8, outCount64); break;
            case 61: unpack<t_ps, 61, t_step>(in8, out8, outCount64); break;
            case 62: unpack<t_ps, 62, t_step>(in8, out8, outCount64); break;
            case 63: unpack<t_ps, 63, t_step>(in8, out8, outCount64); break;
            case 64: unpack<t_ps, 64, t_step>(in8, out8, outCount64); break;
        }
    }
    
    
    
    // ************************************************************************
    // Routines for unpacking and processing
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Interfaces
    // ------------------------------------------------------------------------
    
    // Struct for partial template specialization.
    template<
            processing_style_t t_ps,
            unsigned t_bw,
            unsigned t_step,
            class t_op_processing_unit
    >
    struct unpack_and_process_t {
        static MSV_CXX_ATTRIBUTE_FORCE_INLINE void apply(
                const uint8_t * & in8,
                size_t countIn8,
                typename t_op_processing_unit::state_t & opState
        ) = delete;
    };
    
    // Convenience function.
    template<
            processing_style_t t_ps,
            unsigned t_bw,
            unsigned t_step,
            class t_op_processing_unit
    >
    MSV_CXX_ATTRIBUTE_FORCE_INLINE void unpack_and_process(
            const uint8_t * & in8,
            size_t countIn8,
            typename t_op_processing_unit::state_t & opState
    ) {
        unpack_and_process_t<t_ps, t_bw, t_step, t_op_processing_unit>::apply(
                in8, countIn8, opState
        );
    }
    
    
    // ------------------------------------------------------------------------
    // Template specializations.
    // ------------------------------------------------------------------------
    
    template<unsigned t_bw, class t_op_processing_unit>
    class unpack_and_process_t<processing_style_t::scalar, t_bw, 1, t_op_processing_unit> {
        static const size_t countBits = std::numeric_limits<uint64_t>::digits;
        static const uint64_t mask = bitwidth_max<uint64_t>(t_bw);
        
        struct state_t {
            const uint64_t * in64;
            uint64_t nextOut;
            unsigned bitpos;
            uint64_t tmp;
            
            state_t(const uint64_t * p_In64) {
                in64 = p_In64;
                nextOut = 0;
                // @todo maybe we don't need this
                bitpos = 0;
                tmp = 0;
            }
        };
        
        template<unsigned t_CycleLen, unsigned t_PosInCycle>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void process_block(state_t & s, typename t_op_processing_unit::state_t & opState) {
            if(t_CycleLen > 1) {
                process_block<t_CycleLen / 2, t_PosInCycle                 >(s, opState);
                process_block<t_CycleLen / 2, t_PosInCycle + t_CycleLen / 2>(s, opState);
            }
            else {
                if((t_PosInCycle * t_bw) % countBits == 0) {
                    s.tmp = *(s.in64)++;
                    s.nextOut = mask & s.tmp;
                    s.bitpos = t_bw;
                }
                else if(t_PosInCycle * t_bw / countBits < ((t_PosInCycle + 1) * t_bw - 1) / countBits) {
                    s.tmp = *(s.in64)++;
                    s.nextOut = mask & ((s.tmp << (countBits - s.bitpos + t_bw)) | s.nextOut);
                    s.bitpos = s.bitpos - countBits;
                }
                t_op_processing_unit::apply(s.nextOut, opState);
                s.nextOut = mask & (s.tmp >> s.bitpos);
                s.bitpos += t_bw;
            }
        }
        
    public:
        static void apply(
                const uint8_t * & in8,
                size_t countIn8,
                typename t_op_processing_unit::state_t & opState
        ) {
            const uint64_t * in64 = reinterpret_cast<const uint64_t *>(in8);
            const uint64_t * const endIn64 = in64 + convert_size<uint8_t, uint64_t>(countIn8);
            state_t s(in64);
            while(s.in64 < endIn64)
                process_block<countBits, 0>(s, opState);
        }
    };
    
    // ------------------------------------------------------------------------
    // Selection of the right routine at run-time.
    // ------------------------------------------------------------------------
    
    template<
            processing_style_t t_ps,
            unsigned t_step,
            class t_op_processing_unit
    >
    MSV_CXX_ATTRIBUTE_FORCE_INLINE void unpack_and_process_switch(
            unsigned bitwidth,
            const uint8_t * & in8,
            size_t countIn8,
            typename t_op_processing_unit::state_t & opState
    ) {
        switch(bitwidth) {
            // Generated with Python:
            // for bw in range(1, 64+1):
            //   print("case {: >2}: unpack_and_process<t_ps, {: >2}, t_step, t_op_processing_unit>(in8, countIn8, opState); break;".format(bw, bw))
            case  1: unpack_and_process<t_ps,  1, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case  2: unpack_and_process<t_ps,  2, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case  3: unpack_and_process<t_ps,  3, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case  4: unpack_and_process<t_ps,  4, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case  5: unpack_and_process<t_ps,  5, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case  6: unpack_and_process<t_ps,  6, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case  7: unpack_and_process<t_ps,  7, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case  8: unpack_and_process<t_ps,  8, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case  9: unpack_and_process<t_ps,  9, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 10: unpack_and_process<t_ps, 10, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 11: unpack_and_process<t_ps, 11, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 12: unpack_and_process<t_ps, 12, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 13: unpack_and_process<t_ps, 13, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 14: unpack_and_process<t_ps, 14, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 15: unpack_and_process<t_ps, 15, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 16: unpack_and_process<t_ps, 16, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 17: unpack_and_process<t_ps, 17, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 18: unpack_and_process<t_ps, 18, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 19: unpack_and_process<t_ps, 19, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 20: unpack_and_process<t_ps, 20, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 21: unpack_and_process<t_ps, 21, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 22: unpack_and_process<t_ps, 22, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 23: unpack_and_process<t_ps, 23, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 24: unpack_and_process<t_ps, 24, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 25: unpack_and_process<t_ps, 25, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 26: unpack_and_process<t_ps, 26, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 27: unpack_and_process<t_ps, 27, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 28: unpack_and_process<t_ps, 28, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 29: unpack_and_process<t_ps, 29, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 30: unpack_and_process<t_ps, 30, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 31: unpack_and_process<t_ps, 31, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 32: unpack_and_process<t_ps, 32, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 33: unpack_and_process<t_ps, 33, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 34: unpack_and_process<t_ps, 34, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 35: unpack_and_process<t_ps, 35, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 36: unpack_and_process<t_ps, 36, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 37: unpack_and_process<t_ps, 37, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 38: unpack_and_process<t_ps, 38, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 39: unpack_and_process<t_ps, 39, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 40: unpack_and_process<t_ps, 40, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 41: unpack_and_process<t_ps, 41, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 42: unpack_and_process<t_ps, 42, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 43: unpack_and_process<t_ps, 43, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 44: unpack_and_process<t_ps, 44, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 45: unpack_and_process<t_ps, 45, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 46: unpack_and_process<t_ps, 46, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 47: unpack_and_process<t_ps, 47, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 48: unpack_and_process<t_ps, 48, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 49: unpack_and_process<t_ps, 49, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 50: unpack_and_process<t_ps, 50, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 51: unpack_and_process<t_ps, 51, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 52: unpack_and_process<t_ps, 52, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 53: unpack_and_process<t_ps, 53, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 54: unpack_and_process<t_ps, 54, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 55: unpack_and_process<t_ps, 55, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 56: unpack_and_process<t_ps, 56, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 57: unpack_and_process<t_ps, 57, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 58: unpack_and_process<t_ps, 58, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 59: unpack_and_process<t_ps, 59, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 60: unpack_and_process<t_ps, 60, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 61: unpack_and_process<t_ps, 61, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 62: unpack_and_process<t_ps, 62, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 63: unpack_and_process<t_ps, 63, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
            case 64: unpack_and_process<t_ps, 64, t_step, t_op_processing_unit>(in8, countIn8, opState); break;
        }
    }
}

#endif //MORPHSTORE_CORE_MORPHING_VBP_ROUTINES_H