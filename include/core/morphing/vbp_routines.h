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
 * @todo Somehow include the name of the layout into the way to access these
 *       routines (namespace, struct, name prefix, ...), because we will have
 *       other layouts in the future.
 * @todo Documentation.
 */

#ifndef MORPHSTORE_CORE_MORPHING_VBP_ROUTINES_H
#define MORPHSTORE_CORE_MORPHING_VBP_ROUTINES_H

#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>
#include <vector/primitives/calc.h>
#include <vector/primitives/create.h>
#include <vector/primitives/io.h>
#include <vector/primitives/logic.h>
// @todo The following includes from the vector-lib should not be necessary, I think.
#include <vector/scalar/extension_scalar.h>

#include <cstdint>
#include <immintrin.h>
#include <limits>

/**
 * If this macro is defined, then the functions delegating to the right
 * (un)packing routine for a given bit width, i.e., `pack_switch`,
 * `unpack_switch`, and `unpack_and_process_switch` have a default case that throws an exception if
 * the specified bit width is invalid. For performance reasons, this check
 * should be left out, but during debugging, it can be quite helpful.
 */
#undef VBP_ROUTINE_SWITCH_CHECK_BITWIDTH
#ifdef VBP_ROUTINE_SWITCH_CHECK_BITWIDTH
#include <stdexcept>
#endif

/**
 * This pseudo-bit width can be used to indicate to unpack_switch and
 * unpack_and_process_switch that no action shall be performed. We need this
 * for the non-existent last blocks of the possible incomplete last page of
 * dynamic_vbp_f.
 */
#define VBP_BW_NOBLOCK 0xff

/**
 * If this macro is defined, then the routines generated via template recursion
 * and forced inlining process only the minimum number of input/output vectors
 * required to guarantee that only full vectors are stored/loaded. If it is not
 * defined, then 64 (= number of digits of an uncompressed data element)
 * vectors will be processed in any case.
 */
#define VBP_USE_MIN_CYCLE_LEN

namespace morphstore {
    
    // ************************************************************************
    // Special utilities
    // ************************************************************************
    
    constexpr unsigned minimum_cycle_len(unsigned p_Bw) {
        unsigned cycleLen = std::numeric_limits<uint64_t>::digits;
        while(!(p_Bw & 1)) {
            p_Bw >>= 1;
            cycleLen >>= 1;
        }
        return cycleLen;
    }
    
    
    // ************************************************************************
    // Packing routines
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Interfaces
    // ------------------------------------------------------------------------
    
    // Struct for partial template specialization.
    template<
            class t_vector_extension,
            unsigned t_bw,
            unsigned t_step
    >
    struct pack_t {
        static /*MSV_CXX_ATTRIBUTE_FORCE_INLINE*/ void apply(
                const uint8_t * & in8,
                size_t countIn64,
                uint8_t * & out8
        ) = delete;
    };
    
    // Convenience function.
    template<
            class t_vector_extension,
            unsigned t_bw,
            unsigned t_step
    >
    /*MSV_CXX_ATTRIBUTE_FORCE_INLINE*/ void pack(
            const uint8_t * & in8,
            size_t countIn64,
            uint8_t * & out8
    ) {
        pack_t<t_vector_extension, t_bw, t_step>::apply(in8, countIn64, out8);
    }
    
    
    // ------------------------------------------------------------------------
    // Template specializations.
    // ------------------------------------------------------------------------
    
#if 0
    // Hand-written scalar.
    
    template<unsigned t_bw>
    class pack_t<vector::scalar<vector::v64<uint64_t>>, t_bw, 1> {
        static const size_t countBits = std::numeric_limits<uint64_t>::digits;
        
        struct state_t {
            const uint64_t * in64;
            uint64_t * out64;
            unsigned bitpos;
            uint64_t tmp;
            
            state_t(const uint64_t * p_In64, uint64_t * p_Out64) {
                in64 = p_In64;
                out64 = p_Out64;
                // @todo Maybe we don't need this.
                bitpos = 0;
                tmp = 0;
            }
        };
        
        template<unsigned t_CycleLen, unsigned t_PosInCycle>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void pack_block(state_t & s) {
            if(t_CycleLen > 1) {
                pack_block<t_CycleLen / 2, t_PosInCycle                 >(s);
                pack_block<t_CycleLen / 2, t_PosInCycle + t_CycleLen / 2>(s);
            }
            else {
                const uint64_t tmp2 = *(s.in64)++;
                s.tmp |= (tmp2 << s.bitpos);
                s.bitpos += t_bw;
                if(((t_PosInCycle + 1) * t_bw) % countBits == 0) {
                    *s.out64++ = s.tmp;
                    s.tmp = 0;
                    s.bitpos = 0;
                }
                else if(t_PosInCycle * t_bw / countBits < ((t_PosInCycle + 1) * t_bw - 1) / countBits) {
                    *s.out64++ = s.tmp;
                    s.tmp = tmp2 >> (t_bw - s.bitpos + countBits);
                    s.bitpos -= countBits;
                }
            }
        }
        
    public:
        static void apply(
                const uint8_t * & in8,
                size_t countIn64,
                uint8_t * & out8
        ) {
            const uint64_t * in64 = reinterpret_cast<const uint64_t *>(in8);
            uint64_t * out64 = reinterpret_cast<uint64_t *>(out8);
            state_t s(in64, out64);
            for(unsigned i = 0; i < countIn64; i += 64)
                pack_block<countBits, 0>(s);
            
            in8 = reinterpret_cast<const uint8_t *>(s.in64);
            out8 = reinterpret_cast<uint8_t *>(s.out64);
        }
    };
#else
    // Generic with vector-lib.
    
    template<class t_vector_extension, unsigned t_bw>
    class pack_t<t_vector_extension, t_bw, t_vector_extension::vector_helper_t::element_count::value> {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        static const size_t countBits = std::numeric_limits<base_t>::digits;
        
        struct state_t {
            const base_t * inBase;
            base_t * outBase;
            unsigned bitpos;
            vector_t tmp;
            
            state_t(const base_t * p_InBase, base_t * p_OutBase) {
                inBase = p_InBase;
                outBase = p_OutBase;
                // @todo Maybe we don't need this.
                bitpos = 0;
                tmp = vector::set1<t_ve, vector_base_t_granularity::value>(0);
            }
        };
        
        template<unsigned t_CycleLen, unsigned t_PosInCycle>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void pack_block(state_t & s) {
            using namespace vector;
            if(t_CycleLen > 1) {
                pack_block<t_CycleLen / 2, t_PosInCycle                 >(s);
                pack_block<t_CycleLen / 2, t_PosInCycle + t_CycleLen / 2>(s);
            }
            else {
                const vector_t tmp2 = load<t_ve, iov::ALIGNED, vector_size_bit::value>(s.inBase);
                s.inBase += vector_element_count::value;
                s.tmp = bitwise_or<t_ve>(s.tmp, shift_left<t_ve>::apply(tmp2, s.bitpos));
                s.bitpos += t_bw;
                if(((t_PosInCycle + 1) * t_bw) % countBits == 0) {
                    store<t_ve, iov::ALIGNED, vector_size_bit::value>(s.outBase, s.tmp);
                    s.outBase += vector_element_count::value;
                    s.tmp = set1<t_ve, vector_base_t_granularity::value>(0);
                    s.bitpos = 0;
                }
                else if(t_PosInCycle * t_bw / countBits < ((t_PosInCycle + 1) * t_bw - 1) / countBits) {
                    store<t_ve, iov::ALIGNED, vector_size_bit::value>(s.outBase, s.tmp);
                    s.outBase += vector_element_count::value;
                    s.tmp = shift_right<t_ve>::apply(tmp2, t_bw - s.bitpos + countBits);
                    s.bitpos -= countBits;
                }
            }
        }
        
    public:
        static void apply(
                const uint8_t * & in8,
                size_t countIn64,
                uint8_t * & out8
        ) {
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            base_t * outBase = reinterpret_cast<base_t *>(out8);
            state_t s(inBase, outBase);
            const size_t countInBase = convert_size<uint64_t, base_t>(countIn64);
            
#ifdef VBP_USE_MIN_CYCLE_LEN
            const unsigned cycleLen = minimum_cycle_len(t_bw);
            for(size_t i = 0; i < countInBase; i += cycleLen)
                pack_block<cycleLen, 0>(s);
#else
            const size_t blockSize = vector_size_bit::value;
            for(size_t i = 0; i < countInBase; i += blockSize)
                pack_block<countBits, 0>(s);
#endif
            
            in8 = reinterpret_cast<const uint8_t *>(s.inBase);
            out8 = reinterpret_cast<uint8_t *>(s.outBase);
        }
    };
#endif
    
    // ------------------------------------------------------------------------
    // Selection of the right routine at run-time.
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
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
            //   print("case {: >2}: pack<t_vector_extension, {: >2}, t_step>(in8, inCount64, out8); break;".format(bw, bw))
            case  1: pack<t_vector_extension,  1, t_step>(in8, inCount64, out8); break;
            case  2: pack<t_vector_extension,  2, t_step>(in8, inCount64, out8); break;
            case  3: pack<t_vector_extension,  3, t_step>(in8, inCount64, out8); break;
            case  4: pack<t_vector_extension,  4, t_step>(in8, inCount64, out8); break;
            case  5: pack<t_vector_extension,  5, t_step>(in8, inCount64, out8); break;
            case  6: pack<t_vector_extension,  6, t_step>(in8, inCount64, out8); break;
            case  7: pack<t_vector_extension,  7, t_step>(in8, inCount64, out8); break;
            case  8: pack<t_vector_extension,  8, t_step>(in8, inCount64, out8); break;
            case  9: pack<t_vector_extension,  9, t_step>(in8, inCount64, out8); break;
            case 10: pack<t_vector_extension, 10, t_step>(in8, inCount64, out8); break;
            case 11: pack<t_vector_extension, 11, t_step>(in8, inCount64, out8); break;
            case 12: pack<t_vector_extension, 12, t_step>(in8, inCount64, out8); break;
            case 13: pack<t_vector_extension, 13, t_step>(in8, inCount64, out8); break;
            case 14: pack<t_vector_extension, 14, t_step>(in8, inCount64, out8); break;
            case 15: pack<t_vector_extension, 15, t_step>(in8, inCount64, out8); break;
            case 16: pack<t_vector_extension, 16, t_step>(in8, inCount64, out8); break;
            case 17: pack<t_vector_extension, 17, t_step>(in8, inCount64, out8); break;
            case 18: pack<t_vector_extension, 18, t_step>(in8, inCount64, out8); break;
            case 19: pack<t_vector_extension, 19, t_step>(in8, inCount64, out8); break;
            case 20: pack<t_vector_extension, 20, t_step>(in8, inCount64, out8); break;
            case 21: pack<t_vector_extension, 21, t_step>(in8, inCount64, out8); break;
            case 22: pack<t_vector_extension, 22, t_step>(in8, inCount64, out8); break;
            case 23: pack<t_vector_extension, 23, t_step>(in8, inCount64, out8); break;
            case 24: pack<t_vector_extension, 24, t_step>(in8, inCount64, out8); break;
            case 25: pack<t_vector_extension, 25, t_step>(in8, inCount64, out8); break;
            case 26: pack<t_vector_extension, 26, t_step>(in8, inCount64, out8); break;
            case 27: pack<t_vector_extension, 27, t_step>(in8, inCount64, out8); break;
            case 28: pack<t_vector_extension, 28, t_step>(in8, inCount64, out8); break;
            case 29: pack<t_vector_extension, 29, t_step>(in8, inCount64, out8); break;
            case 30: pack<t_vector_extension, 30, t_step>(in8, inCount64, out8); break;
            case 31: pack<t_vector_extension, 31, t_step>(in8, inCount64, out8); break;
            case 32: pack<t_vector_extension, 32, t_step>(in8, inCount64, out8); break;
            case 33: pack<t_vector_extension, 33, t_step>(in8, inCount64, out8); break;
            case 34: pack<t_vector_extension, 34, t_step>(in8, inCount64, out8); break;
            case 35: pack<t_vector_extension, 35, t_step>(in8, inCount64, out8); break;
            case 36: pack<t_vector_extension, 36, t_step>(in8, inCount64, out8); break;
            case 37: pack<t_vector_extension, 37, t_step>(in8, inCount64, out8); break;
            case 38: pack<t_vector_extension, 38, t_step>(in8, inCount64, out8); break;
            case 39: pack<t_vector_extension, 39, t_step>(in8, inCount64, out8); break;
            case 40: pack<t_vector_extension, 40, t_step>(in8, inCount64, out8); break;
            case 41: pack<t_vector_extension, 41, t_step>(in8, inCount64, out8); break;
            case 42: pack<t_vector_extension, 42, t_step>(in8, inCount64, out8); break;
            case 43: pack<t_vector_extension, 43, t_step>(in8, inCount64, out8); break;
            case 44: pack<t_vector_extension, 44, t_step>(in8, inCount64, out8); break;
            case 45: pack<t_vector_extension, 45, t_step>(in8, inCount64, out8); break;
            case 46: pack<t_vector_extension, 46, t_step>(in8, inCount64, out8); break;
            case 47: pack<t_vector_extension, 47, t_step>(in8, inCount64, out8); break;
            case 48: pack<t_vector_extension, 48, t_step>(in8, inCount64, out8); break;
            case 49: pack<t_vector_extension, 49, t_step>(in8, inCount64, out8); break;
            case 50: pack<t_vector_extension, 50, t_step>(in8, inCount64, out8); break;
            case 51: pack<t_vector_extension, 51, t_step>(in8, inCount64, out8); break;
            case 52: pack<t_vector_extension, 52, t_step>(in8, inCount64, out8); break;
            case 53: pack<t_vector_extension, 53, t_step>(in8, inCount64, out8); break;
            case 54: pack<t_vector_extension, 54, t_step>(in8, inCount64, out8); break;
            case 55: pack<t_vector_extension, 55, t_step>(in8, inCount64, out8); break;
            case 56: pack<t_vector_extension, 56, t_step>(in8, inCount64, out8); break;
            case 57: pack<t_vector_extension, 57, t_step>(in8, inCount64, out8); break;
            case 58: pack<t_vector_extension, 58, t_step>(in8, inCount64, out8); break;
            case 59: pack<t_vector_extension, 59, t_step>(in8, inCount64, out8); break;
            case 60: pack<t_vector_extension, 60, t_step>(in8, inCount64, out8); break;
            case 61: pack<t_vector_extension, 61, t_step>(in8, inCount64, out8); break;
            case 62: pack<t_vector_extension, 62, t_step>(in8, inCount64, out8); break;
            case 63: pack<t_vector_extension, 63, t_step>(in8, inCount64, out8); break;
            case 64: pack<t_vector_extension, 64, t_step>(in8, inCount64, out8); break;
            // Packing does not require the case for VBP_BW_NOBLOCK.
#ifdef VBP_ROUTINE_SWITCH_CHECK_BITWIDTH
            default: throw std::runtime_error(
                    "pack_switch: unsupported bit width: " +
                    std::to_string(bitwidth)
            );
#endif
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
            class t_vector_extension,
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
            class t_vector_extension,
            unsigned t_bw,
            unsigned t_step
    >
    MSV_CXX_ATTRIBUTE_FORCE_INLINE void unpack(
            const uint8_t * & in8,
            uint8_t * & out8,
            size_t countOut64
    ) {
        unpack_t<t_vector_extension, t_bw, t_step>::apply(in8, out8, countOut64);
    }
    
    
    // ------------------------------------------------------------------------
    // Template specializations.
    // ------------------------------------------------------------------------
    
#if 0
    // Hand-written scalar.
    
    template<unsigned t_bw>
    class unpack_t<vector::scalar<vector::v64<uint64_t>>, t_bw, 1> {
        static const size_t countBits = std::numeric_limits<uint64_t>::digits;
        static const uint64_t mask = bitwidth_max<uint64_t>(t_bw);
        
        struct state_t {
            const uint64_t * in64;
            uint64_t * out64;
            uint64_t nextOut;
            unsigned bitpos;
            uint64_t tmp;
            
            state_t(const uint64_t * p_In64, uint64_t * p_Out64) {
                in64 = p_In64;
                out64 = p_Out64;
                nextOut = 0;
                // @todo Maybe we don't need this.
                bitpos = 0;
                tmp = 0;
            }
        };
        
        template<unsigned t_CycleLen, unsigned t_PosInCycle>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void unpack_block(state_t & s) {
            if(t_CycleLen > 1) {
                unpack_block<t_CycleLen / 2, t_PosInCycle                 >(s);
                unpack_block<t_CycleLen / 2, t_PosInCycle + t_CycleLen / 2>(s);
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
                *(s.out64)++ = s.nextOut;
                s.nextOut = mask & (s.tmp >> s.bitpos);
                s.bitpos += t_bw;
            }
        }
        
    public:
        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countOut64
        ) {
            const uint64_t * in64 = reinterpret_cast<const uint64_t *>(in8);
            uint64_t * out64 = reinterpret_cast<uint64_t *>(out8);
            state_t s(in64, out64);
            for(unsigned i = 0; i < countOut64; i += 64)
                unpack_block<countBits, 0>(s);
            
            in8 = reinterpret_cast<const uint8_t *>(s.in64);
            out8 = reinterpret_cast<uint8_t *>(s.out64);
        }
    };
#else
    // Generic with vector-lib.
    
    template<class t_vector_extension, unsigned t_bw>
    class unpack_t<t_vector_extension, t_bw, t_vector_extension::vector_helper_t::element_count::value> {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        static const size_t countBits = std::numeric_limits<base_t>::digits;
        // @todo It would be nice to initialize this in-class. However, the
        // compiler complains because set1 is not constexpr, even when it is
        // defined so.
        static const vector_t mask; // = vector::set1<t_ve, vector_base_t_granularity::value>(bitwidth_max<base_t>(t_bw));
        
        struct state_t {
            const base_t * inBase;
            base_t * outBase;
            vector_t nextOut;
            unsigned bitpos;
            vector_t tmp;
            
            state_t(const base_t * p_InBase, base_t * p_OutBase) {
                inBase = p_InBase;
                outBase = p_OutBase;
                nextOut = vector::set1<t_ve, vector_base_t_granularity::value>(0);
                // @todo Maybe we don't need this.
                bitpos = 0;
                tmp = vector::set1<t_ve, vector_base_t_granularity::value>(0);
            }
        };
        
        template<unsigned t_CycleLen, unsigned t_PosInCycle>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void unpack_block(state_t & s) {
            using namespace vector;
            
            if(t_CycleLen > 1) {
                unpack_block<t_CycleLen / 2, t_PosInCycle                 >(s);
                unpack_block<t_CycleLen / 2, t_PosInCycle + t_CycleLen / 2>(s);
            }
            else {
                if((t_PosInCycle * t_bw) % countBits == 0) {
                    s.tmp = load<t_ve, iov::ALIGNED, vector_size_bit::value>(s.inBase);
                    s.inBase += vector_element_count::value;
                    s.nextOut = bitwise_and<t_ve>(mask, s.tmp);
                    s.bitpos = t_bw;
                }
                else if(t_PosInCycle * t_bw / countBits < ((t_PosInCycle + 1) * t_bw - 1) / countBits) {
                    s.tmp = load<t_ve, iov::ALIGNED, vector_size_bit::value>(s.inBase);
                    s.inBase += vector_element_count::value;
                    s.nextOut = bitwise_and<t_ve>(mask, bitwise_or<t_ve>(shift_left<t_ve>::apply(s.tmp, countBits - s.bitpos + t_bw), s.nextOut));
                    s.bitpos = s.bitpos - countBits;
                }
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(s.outBase, s.nextOut);
                s.outBase += vector_element_count::value;
                s.nextOut = bitwise_and<t_ve>(mask, shift_right<t_ve>::apply(s.tmp, s.bitpos));
                s.bitpos += t_bw;
            }
        }
        
    public:
        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countOut64
        ) {
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            base_t * outBase = reinterpret_cast<base_t *>(out8);
            state_t s(inBase, outBase);
            const size_t countOutBase = convert_size<uint64_t, base_t>(countOut64);
#ifdef VBP_USE_MIN_CYCLE_LEN
            const unsigned cycleLen = minimum_cycle_len(t_bw);
            for(size_t i = 0; i < countOutBase; i += cycleLen)
                unpack_block<cycleLen, 0>(s);
#else
            const size_t blockSize = vector_size_bit::value;
            for(size_t i = 0; i < countOutBase; i += blockSize)
                unpack_block<countBits, 0>(s);
#endif
            
            in8 = reinterpret_cast<const uint8_t *>(s.inBase);
            out8 = reinterpret_cast<uint8_t *>(s.outBase);
        }
    };
    
    template<class t_vector_extension, unsigned t_bw>
    const typename t_vector_extension::vector_t unpack_t<
            t_vector_extension,
            t_bw,
            t_vector_extension::vector_helper_t::element_count::value
    >::mask = vector::set1<
            t_vector_extension,
            t_vector_extension::vector_helper_t::granularity::value
    >(
            bitwidth_max<typename t_vector_extension::base_t>(t_bw)
    );
#endif
    
    // ------------------------------------------------------------------------
    // Selection of the right routine at run-time.
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
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
            //   print("case {: >2}: unpack<t_vector_extension, {: >2}, t_step>(in8, out8, outCount64); break;".format(bw, bw))
            case  1: unpack<t_vector_extension,  1, t_step>(in8, out8, outCount64); break;
            case  2: unpack<t_vector_extension,  2, t_step>(in8, out8, outCount64); break;
            case  3: unpack<t_vector_extension,  3, t_step>(in8, out8, outCount64); break;
            case  4: unpack<t_vector_extension,  4, t_step>(in8, out8, outCount64); break;
            case  5: unpack<t_vector_extension,  5, t_step>(in8, out8, outCount64); break;
            case  6: unpack<t_vector_extension,  6, t_step>(in8, out8, outCount64); break;
            case  7: unpack<t_vector_extension,  7, t_step>(in8, out8, outCount64); break;
            case  8: unpack<t_vector_extension,  8, t_step>(in8, out8, outCount64); break;
            case  9: unpack<t_vector_extension,  9, t_step>(in8, out8, outCount64); break;
            case 10: unpack<t_vector_extension, 10, t_step>(in8, out8, outCount64); break;
            case 11: unpack<t_vector_extension, 11, t_step>(in8, out8, outCount64); break;
            case 12: unpack<t_vector_extension, 12, t_step>(in8, out8, outCount64); break;
            case 13: unpack<t_vector_extension, 13, t_step>(in8, out8, outCount64); break;
            case 14: unpack<t_vector_extension, 14, t_step>(in8, out8, outCount64); break;
            case 15: unpack<t_vector_extension, 15, t_step>(in8, out8, outCount64); break;
            case 16: unpack<t_vector_extension, 16, t_step>(in8, out8, outCount64); break;
            case 17: unpack<t_vector_extension, 17, t_step>(in8, out8, outCount64); break;
            case 18: unpack<t_vector_extension, 18, t_step>(in8, out8, outCount64); break;
            case 19: unpack<t_vector_extension, 19, t_step>(in8, out8, outCount64); break;
            case 20: unpack<t_vector_extension, 20, t_step>(in8, out8, outCount64); break;
            case 21: unpack<t_vector_extension, 21, t_step>(in8, out8, outCount64); break;
            case 22: unpack<t_vector_extension, 22, t_step>(in8, out8, outCount64); break;
            case 23: unpack<t_vector_extension, 23, t_step>(in8, out8, outCount64); break;
            case 24: unpack<t_vector_extension, 24, t_step>(in8, out8, outCount64); break;
            case 25: unpack<t_vector_extension, 25, t_step>(in8, out8, outCount64); break;
            case 26: unpack<t_vector_extension, 26, t_step>(in8, out8, outCount64); break;
            case 27: unpack<t_vector_extension, 27, t_step>(in8, out8, outCount64); break;
            case 28: unpack<t_vector_extension, 28, t_step>(in8, out8, outCount64); break;
            case 29: unpack<t_vector_extension, 29, t_step>(in8, out8, outCount64); break;
            case 30: unpack<t_vector_extension, 30, t_step>(in8, out8, outCount64); break;
            case 31: unpack<t_vector_extension, 31, t_step>(in8, out8, outCount64); break;
            case 32: unpack<t_vector_extension, 32, t_step>(in8, out8, outCount64); break;
            case 33: unpack<t_vector_extension, 33, t_step>(in8, out8, outCount64); break;
            case 34: unpack<t_vector_extension, 34, t_step>(in8, out8, outCount64); break;
            case 35: unpack<t_vector_extension, 35, t_step>(in8, out8, outCount64); break;
            case 36: unpack<t_vector_extension, 36, t_step>(in8, out8, outCount64); break;
            case 37: unpack<t_vector_extension, 37, t_step>(in8, out8, outCount64); break;
            case 38: unpack<t_vector_extension, 38, t_step>(in8, out8, outCount64); break;
            case 39: unpack<t_vector_extension, 39, t_step>(in8, out8, outCount64); break;
            case 40: unpack<t_vector_extension, 40, t_step>(in8, out8, outCount64); break;
            case 41: unpack<t_vector_extension, 41, t_step>(in8, out8, outCount64); break;
            case 42: unpack<t_vector_extension, 42, t_step>(in8, out8, outCount64); break;
            case 43: unpack<t_vector_extension, 43, t_step>(in8, out8, outCount64); break;
            case 44: unpack<t_vector_extension, 44, t_step>(in8, out8, outCount64); break;
            case 45: unpack<t_vector_extension, 45, t_step>(in8, out8, outCount64); break;
            case 46: unpack<t_vector_extension, 46, t_step>(in8, out8, outCount64); break;
            case 47: unpack<t_vector_extension, 47, t_step>(in8, out8, outCount64); break;
            case 48: unpack<t_vector_extension, 48, t_step>(in8, out8, outCount64); break;
            case 49: unpack<t_vector_extension, 49, t_step>(in8, out8, outCount64); break;
            case 50: unpack<t_vector_extension, 50, t_step>(in8, out8, outCount64); break;
            case 51: unpack<t_vector_extension, 51, t_step>(in8, out8, outCount64); break;
            case 52: unpack<t_vector_extension, 52, t_step>(in8, out8, outCount64); break;
            case 53: unpack<t_vector_extension, 53, t_step>(in8, out8, outCount64); break;
            case 54: unpack<t_vector_extension, 54, t_step>(in8, out8, outCount64); break;
            case 55: unpack<t_vector_extension, 55, t_step>(in8, out8, outCount64); break;
            case 56: unpack<t_vector_extension, 56, t_step>(in8, out8, outCount64); break;
            case 57: unpack<t_vector_extension, 57, t_step>(in8, out8, outCount64); break;
            case 58: unpack<t_vector_extension, 58, t_step>(in8, out8, outCount64); break;
            case 59: unpack<t_vector_extension, 59, t_step>(in8, out8, outCount64); break;
            case 60: unpack<t_vector_extension, 60, t_step>(in8, out8, outCount64); break;
            case 61: unpack<t_vector_extension, 61, t_step>(in8, out8, outCount64); break;
            case 62: unpack<t_vector_extension, 62, t_step>(in8, out8, outCount64); break;
            case 63: unpack<t_vector_extension, 63, t_step>(in8, out8, outCount64); break;
            case 64: unpack<t_vector_extension, 64, t_step>(in8, out8, outCount64); break;
            case VBP_BW_NOBLOCK: /* do nothing */ break;
#ifdef VBP_ROUTINE_SWITCH_CHECK_BITWIDTH
            default: throw std::runtime_error(
                    "unpack_switch: unsupported bit width: " +
                    std::to_string(bitwidth)
            );
#endif
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
            class t_vector_extension,
            unsigned t_bw,
            unsigned t_step,
            template<class, class ...> class t_op_processing_unit,
            class ... t_extra_args
    >
    struct unpack_and_process_t {
        static MSV_CXX_ATTRIBUTE_FORCE_INLINE void apply(
                const uint8_t * & in8,
                size_t countIn8,
                typename t_op_processing_unit<
                        t_vector_extension,
                        t_extra_args ...
                >::state_t & opState
        ) = delete;
    };
    
    // Convenience function.
    template<
            class t_vector_extension,
            unsigned t_bw,
            unsigned t_step,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    MSV_CXX_ATTRIBUTE_FORCE_INLINE void unpack_and_process(
            const uint8_t * & in8,
            size_t countIn8,
            typename t_op_vector<
                    t_vector_extension,
                    t_extra_args ...
            >::state_t & opState
    ) {
        unpack_and_process_t<
                t_vector_extension, t_bw, t_step, t_op_vector, t_extra_args ...
        >::apply(
                in8, countIn8, opState
        );
    }
    
    
    // ------------------------------------------------------------------------
    // Template specializations.
    // ------------------------------------------------------------------------
    
#if 0
    // Hand-written scalar
    
    template<unsigned t_bw, template<class /*t_vector_extension*/> class t_op_processing_unit>
    class unpack_and_process_t<vector::scalar<vector::v64<uint64_t>>, t_bw, 1, t_op_processing_unit> {
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
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void unpack_and_process_block(
                state_t & s,
                typename t_op_processing_unit<vector::scalar<vector::v64<uint64_t>>>::state_t & opState
        ) {
            if(t_CycleLen > 1) {
                unpack_and_process_block<t_CycleLen / 2, t_PosInCycle                 >(s, opState);
                unpack_and_process_block<t_CycleLen / 2, t_PosInCycle + t_CycleLen / 2>(s, opState);
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
                t_op_processing_unit<vector::scalar<vector::v64<uint64_t>>>::apply(s.nextOut, opState);
                s.nextOut = mask & (s.tmp >> s.bitpos);
                s.bitpos += t_bw;
            }
        }
        
    public:
        static void apply(
                const uint8_t * & in8,
                size_t countIn8,
                typename t_op_processing_unit<vector::scalar<vector::v64<uint64_t>>>::state_t & opState
        ) {
            const uint64_t * in64 = reinterpret_cast<const uint64_t *>(in8);
            const uint64_t * const endIn64 = in64 + convert_size<uint8_t, uint64_t>(countIn8);
            state_t s(in64);
            while(s.in64 < endIn64)
                unpack_and_process_block<countBits, 0>(s, opState);
        }
    };
#else
    // Generic with vector-lib.
    
    template<
            class t_vector_extension,
            unsigned t_bw,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    class unpack_and_process_t<
            t_vector_extension,
            t_bw,
            t_vector_extension::vector_helper_t::element_count::value,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        static const size_t countBits = std::numeric_limits<base_t>::digits;
        // @todo It would be nice to initialize this in-class. However, the
        // compiler complains because set1 is not constexpr, even when it is
        // defined so.
        static const vector_t mask; // = vector::set1<t_ve, vector_base_t_granularity::value>(bitwidth_max<base_t>(t_bw));
        
        struct state_t {
            const base_t * inBase;
            vector_t nextOut;
            unsigned bitpos;
            vector_t tmp;
            
            state_t(const base_t * p_InBase) {
                inBase = p_InBase;
                nextOut = vector::set1<t_ve, vector_base_t_granularity::value>(0);
                // @todo Maybe we don't need this.
                bitpos = 0;
                tmp = vector::set1<t_ve, vector_base_t_granularity::value>(0);
            }
        };
        
        template<unsigned t_CycleLen, unsigned t_PosInCycle>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void unpack_and_process_block(
                state_t & s,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & opState
        ) {
            using namespace vector;
            
            if(t_CycleLen > 1) {
                unpack_and_process_block<t_CycleLen / 2, t_PosInCycle                 >(s, opState);
                unpack_and_process_block<t_CycleLen / 2, t_PosInCycle + t_CycleLen / 2>(s, opState);
            }
            else {
                if((t_PosInCycle * t_bw) % countBits == 0) {
                    s.tmp = load<t_ve, iov::ALIGNED, vector_size_bit::value>(s.inBase);
                    s.inBase += vector_element_count::value;
                    s.nextOut = bitwise_and<t_ve>(mask, s.tmp);
                    s.bitpos = t_bw;
                }
                else if(t_PosInCycle * t_bw / countBits < ((t_PosInCycle + 1) * t_bw - 1) / countBits) {
                    s.tmp = load<t_ve, iov::ALIGNED, vector_size_bit::value>(s.inBase);
                    s.inBase += vector_element_count::value;
                    s.nextOut = bitwise_and<t_ve>(mask, bitwise_or<t_ve>(shift_left<t_ve>::apply(s.tmp, countBits - s.bitpos + t_bw), s.nextOut));
                    s.bitpos = s.bitpos - countBits;
                }
                t_op_vector<t_ve, t_extra_args ...>::apply(s.nextOut, opState);
                s.nextOut = bitwise_and<t_ve>(mask, shift_right<t_ve>::apply(s.tmp, s.bitpos));
                s.bitpos += t_bw;
            }
        }
        
    public:
        static void apply(
                const uint8_t * & in8,
                size_t countIn8,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & opState
        ) {
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            const base_t * const endInBase = inBase + convert_size<uint8_t, base_t>(countIn8);
            state_t s(inBase);
#ifdef VBP_USE_MIN_CYCLE_LEN
            const unsigned cycleLen = minimum_cycle_len(t_bw);
            while(s.inBase < endInBase)
                unpack_and_process_block<cycleLen, 0>(s, opState);
#else
            while(s.inBase < endInBase)
                unpack_and_process_block<countBits, 0>(s, opState);
#endif
            
            in8 = reinterpret_cast<const uint8_t *>(s.inBase);
        }
    };
    
    template<
            class t_vector_extension,
            unsigned t_bw,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    const typename t_vector_extension::vector_t unpack_and_process_t<
            t_vector_extension,
            t_bw,
            t_vector_extension::vector_helper_t::element_count::value,
            t_op_vector,
            t_extra_args ...
    >::mask = vector::set1<
            t_vector_extension,
            t_vector_extension::vector_helper_t::granularity::value
    >(
            bitwidth_max<typename t_vector_extension::base_t>(t_bw)
    );
#endif
    
    // ------------------------------------------------------------------------
    // Selection of the right routine at run-time.
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            unsigned t_step,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    MSV_CXX_ATTRIBUTE_FORCE_INLINE void unpack_and_process_switch(
            unsigned bitwidth,
            const uint8_t * & in8,
            size_t countIn8,
            typename t_op_vector<
                    t_vector_extension, t_extra_args ...
            >::state_t & opState
    ) {
        using t_ve = t_vector_extension;
        
        switch(bitwidth) {
            // Generated with Python:
            // for bw in range(1, 64+1):
            //   print("case {: >2}: unpack_and_process<t_ve, {: >2}, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;".format(bw, bw))
            case  1: unpack_and_process<t_ve,  1, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case  2: unpack_and_process<t_ve,  2, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case  3: unpack_and_process<t_ve,  3, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case  4: unpack_and_process<t_ve,  4, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case  5: unpack_and_process<t_ve,  5, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case  6: unpack_and_process<t_ve,  6, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case  7: unpack_and_process<t_ve,  7, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case  8: unpack_and_process<t_ve,  8, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case  9: unpack_and_process<t_ve,  9, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 10: unpack_and_process<t_ve, 10, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 11: unpack_and_process<t_ve, 11, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 12: unpack_and_process<t_ve, 12, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 13: unpack_and_process<t_ve, 13, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 14: unpack_and_process<t_ve, 14, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 15: unpack_and_process<t_ve, 15, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 16: unpack_and_process<t_ve, 16, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 17: unpack_and_process<t_ve, 17, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 18: unpack_and_process<t_ve, 18, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 19: unpack_and_process<t_ve, 19, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 20: unpack_and_process<t_ve, 20, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 21: unpack_and_process<t_ve, 21, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 22: unpack_and_process<t_ve, 22, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 23: unpack_and_process<t_ve, 23, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 24: unpack_and_process<t_ve, 24, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 25: unpack_and_process<t_ve, 25, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 26: unpack_and_process<t_ve, 26, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 27: unpack_and_process<t_ve, 27, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 28: unpack_and_process<t_ve, 28, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 29: unpack_and_process<t_ve, 29, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 30: unpack_and_process<t_ve, 30, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 31: unpack_and_process<t_ve, 31, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 32: unpack_and_process<t_ve, 32, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 33: unpack_and_process<t_ve, 33, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 34: unpack_and_process<t_ve, 34, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 35: unpack_and_process<t_ve, 35, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 36: unpack_and_process<t_ve, 36, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 37: unpack_and_process<t_ve, 37, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 38: unpack_and_process<t_ve, 38, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 39: unpack_and_process<t_ve, 39, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 40: unpack_and_process<t_ve, 40, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 41: unpack_and_process<t_ve, 41, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 42: unpack_and_process<t_ve, 42, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 43: unpack_and_process<t_ve, 43, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 44: unpack_and_process<t_ve, 44, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 45: unpack_and_process<t_ve, 45, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 46: unpack_and_process<t_ve, 46, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 47: unpack_and_process<t_ve, 47, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 48: unpack_and_process<t_ve, 48, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 49: unpack_and_process<t_ve, 49, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 50: unpack_and_process<t_ve, 50, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 51: unpack_and_process<t_ve, 51, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 52: unpack_and_process<t_ve, 52, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 53: unpack_and_process<t_ve, 53, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 54: unpack_and_process<t_ve, 54, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 55: unpack_and_process<t_ve, 55, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 56: unpack_and_process<t_ve, 56, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 57: unpack_and_process<t_ve, 57, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 58: unpack_and_process<t_ve, 58, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 59: unpack_and_process<t_ve, 59, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 60: unpack_and_process<t_ve, 60, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 61: unpack_and_process<t_ve, 61, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 62: unpack_and_process<t_ve, 62, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 63: unpack_and_process<t_ve, 63, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case 64: unpack_and_process<t_ve, 64, t_step, t_op_vector, t_extra_args ...>(in8, countIn8, opState); break;
            case VBP_BW_NOBLOCK: /* do nothing */ break;
#ifdef VBP_ROUTINE_SWITCH_CHECK_BITWIDTH
            default: throw std::runtime_error(
                    "unpack_and_process_switch: unsupported bit width: " +
                    std::to_string(bitwidth)
            );
#endif
        }
    }
}

#undef VBP_ROUTINE_SWITCH_CHECK_BITWIDTH
#undef VBP_USE_MIN_CYCLE_LEN

#endif //MORPHSTORE_CORE_MORPHING_VBP_ROUTINES_H