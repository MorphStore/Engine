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
 * @file vbp_commons.h
 * @brief Things that are useful for multiple variants of the vertical
 * bit-packed layout.
 * @todo Documentation.
 */

#ifndef MORPHSTORE_CORE_MORPHING_VBP_COMMONS_H
#define MORPHSTORE_CORE_MORPHING_VBP_COMMONS_H

#include <core/morphing/morph.h>
#include <core/utils/basic_types.h>
#include <core/utils/preprocessor.h>

#include <cstdint>

/**
 * If this macro is defined, then the functions delegating to the right
 * template specialization of morph_batch for (un)packing data using a given
 * bit width, i.e., `pack_switch`, `unpack_switch`, and
 * `decompress_and_process_batch_switch` have a default case that throws an
 * exception if the specified bit width is invalid. For performance reasons,
 * this check should be left out, but during debugging, it can be quite
 * helpful.
 */
#undef VBP_ROUTINE_SWITCH_CHECK_BITWIDTH
#ifdef VBP_ROUTINE_SWITCH_CHECK_BITWIDTH
#include <stdexcept>
#endif

/**
 * This pseudo-bit width can be used to indicate to `unpack_switch` and
 * `decompress_and_process_batch_switch` that no action shall be performed. We
 * need this for the non-existent last blocks of the possible incomplete last
 * page of `dynamic_vbp_f`.
 */
#define VBP_BW_NOBLOCK 0xff

/**
 * The following macros control whether the respective kind of routine is
 * declared to be always inlined.
 */
#undef VBP_FORCE_INLINE_PACK
#undef VBP_FORCE_INLINE_PACK_SWITCH
#undef VBP_FORCE_INLINE_UNPACK
#undef VBP_FORCE_INLINE_UNPACK_SWITCH
#undef VBP_FORCE_INLINE_UNPACK_AND_PROCESS
#undef VBP_FORCE_INLINE_UNPACK_AND_PROCESS_SWITCH

// @todo Remove this workaround. We should find a cleaner way to do this.
/**
 * If the following macro is defined, only the routines for the bitwidths
 * actually needed for SSB at scale factor 1 are compiled, which reduces the
 * compile time significantly.
 */
#define VBP_LIMIT_ROUTINES_FOR_SSB_SF1

namespace morphstore {
    
    // ************************************************************************
    // Selection of the right routine at run-time.
    // ************************************************************************

    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            template<unsigned, unsigned> class t_dst_l,
            unsigned t_Step
    >
#ifdef VBP_FORCE_INLINE_PACK_SWITCH
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
#endif
    void pack_switch(
            unsigned bitwidth,
            const uint8_t * & in8,
            uint8_t * & out8,
            size_t countLog
    ) {
        switch(bitwidth) {
            // Generated with Python:
            // for bw in range(1, 64+1):
            //   print("case {: >2}: morph_batch<t_vector_extension, t_dst_l<{: >2}, t_Step>, uncompr_f>(in8, out8, countLog); break;".format(bw, bw))
            case  1: morph_batch<t_vector_extension, t_dst_l< 1, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case  2: morph_batch<t_vector_extension, t_dst_l< 2, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case  3: morph_batch<t_vector_extension, t_dst_l< 3, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case  4: morph_batch<t_vector_extension, t_dst_l< 4, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case  5: morph_batch<t_vector_extension, t_dst_l< 5, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case  6: morph_batch<t_vector_extension, t_dst_l< 6, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case  7: morph_batch<t_vector_extension, t_dst_l< 7, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case  8: morph_batch<t_vector_extension, t_dst_l< 8, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case  9: morph_batch<t_vector_extension, t_dst_l< 9, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 10: morph_batch<t_vector_extension, t_dst_l<10, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 11: morph_batch<t_vector_extension, t_dst_l<11, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 12: morph_batch<t_vector_extension, t_dst_l<12, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 13: morph_batch<t_vector_extension, t_dst_l<13, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 14: morph_batch<t_vector_extension, t_dst_l<14, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 15: morph_batch<t_vector_extension, t_dst_l<15, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 16: morph_batch<t_vector_extension, t_dst_l<16, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 17: morph_batch<t_vector_extension, t_dst_l<17, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 18: morph_batch<t_vector_extension, t_dst_l<18, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 19: morph_batch<t_vector_extension, t_dst_l<19, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 20: morph_batch<t_vector_extension, t_dst_l<20, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 21: morph_batch<t_vector_extension, t_dst_l<21, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 22: morph_batch<t_vector_extension, t_dst_l<22, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 23: morph_batch<t_vector_extension, t_dst_l<23, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 24: morph_batch<t_vector_extension, t_dst_l<24, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 25: morph_batch<t_vector_extension, t_dst_l<25, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 26: morph_batch<t_vector_extension, t_dst_l<26, t_Step>, uncompr_f>(in8, out8, countLog); break;
#ifndef VBP_LIMIT_ROUTINES_FOR_SSB_SF1
            case 27: morph_batch<t_vector_extension, t_dst_l<27, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 28: morph_batch<t_vector_extension, t_dst_l<28, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 29: morph_batch<t_vector_extension, t_dst_l<29, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 30: morph_batch<t_vector_extension, t_dst_l<30, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 31: morph_batch<t_vector_extension, t_dst_l<31, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 32: morph_batch<t_vector_extension, t_dst_l<32, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 33: morph_batch<t_vector_extension, t_dst_l<33, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 34: morph_batch<t_vector_extension, t_dst_l<34, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 35: morph_batch<t_vector_extension, t_dst_l<35, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 36: morph_batch<t_vector_extension, t_dst_l<36, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 37: morph_batch<t_vector_extension, t_dst_l<37, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 38: morph_batch<t_vector_extension, t_dst_l<38, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 39: morph_batch<t_vector_extension, t_dst_l<39, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 40: morph_batch<t_vector_extension, t_dst_l<40, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 41: morph_batch<t_vector_extension, t_dst_l<41, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 42: morph_batch<t_vector_extension, t_dst_l<42, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 43: morph_batch<t_vector_extension, t_dst_l<43, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 44: morph_batch<t_vector_extension, t_dst_l<44, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 45: morph_batch<t_vector_extension, t_dst_l<45, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 46: morph_batch<t_vector_extension, t_dst_l<46, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 47: morph_batch<t_vector_extension, t_dst_l<47, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 48: morph_batch<t_vector_extension, t_dst_l<48, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 49: morph_batch<t_vector_extension, t_dst_l<49, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 50: morph_batch<t_vector_extension, t_dst_l<50, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 51: morph_batch<t_vector_extension, t_dst_l<51, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 52: morph_batch<t_vector_extension, t_dst_l<52, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 53: morph_batch<t_vector_extension, t_dst_l<53, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 54: morph_batch<t_vector_extension, t_dst_l<54, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 55: morph_batch<t_vector_extension, t_dst_l<55, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 56: morph_batch<t_vector_extension, t_dst_l<56, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 57: morph_batch<t_vector_extension, t_dst_l<57, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 58: morph_batch<t_vector_extension, t_dst_l<58, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 59: morph_batch<t_vector_extension, t_dst_l<59, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 60: morph_batch<t_vector_extension, t_dst_l<60, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 61: morph_batch<t_vector_extension, t_dst_l<61, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 62: morph_batch<t_vector_extension, t_dst_l<62, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 63: morph_batch<t_vector_extension, t_dst_l<63, t_Step>, uncompr_f>(in8, out8, countLog); break;
            case 64: morph_batch<t_vector_extension, t_dst_l<64, t_Step>, uncompr_f>(in8, out8, countLog); break;
#endif
            // Packing does not require the case for VBP_BW_NOBLOCK.
#ifdef VBP_ROUTINE_SWITCH_CHECK_BITWIDTH
            default: throw std::runtime_error(
                    "pack_switch: unsupported bit width: " +
                    std::to_string(bitwidth)
            );
#endif
        }
    }

    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            template<unsigned, unsigned> class t_src_l,
            unsigned t_Step
    >
#ifdef VBP_FORCE_INLINE_UNPACK_SWITCH
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
#endif
    void unpack_switch(
            unsigned bitwidth,
            const uint8_t * & in8,
            uint8_t * & out8,
            size_t countLog
    ) {
        switch(bitwidth) {
            // Generated with Python:
            // for bw in range(1, 64+1):
            //   print("case {: >2}: morph_batch<t_vector_extension, uncompr_f, t_src_l<{: >2}, t_Step>>(in8, out8, countLog); break;".format(bw, bw))
            case  1: morph_batch<t_vector_extension, uncompr_f, t_src_l< 1, t_Step>>(in8, out8, countLog); break;
            case  2: morph_batch<t_vector_extension, uncompr_f, t_src_l< 2, t_Step>>(in8, out8, countLog); break;
            case  3: morph_batch<t_vector_extension, uncompr_f, t_src_l< 3, t_Step>>(in8, out8, countLog); break;
            case  4: morph_batch<t_vector_extension, uncompr_f, t_src_l< 4, t_Step>>(in8, out8, countLog); break;
            case  5: morph_batch<t_vector_extension, uncompr_f, t_src_l< 5, t_Step>>(in8, out8, countLog); break;
            case  6: morph_batch<t_vector_extension, uncompr_f, t_src_l< 6, t_Step>>(in8, out8, countLog); break;
            case  7: morph_batch<t_vector_extension, uncompr_f, t_src_l< 7, t_Step>>(in8, out8, countLog); break;
            case  8: morph_batch<t_vector_extension, uncompr_f, t_src_l< 8, t_Step>>(in8, out8, countLog); break;
            case  9: morph_batch<t_vector_extension, uncompr_f, t_src_l< 9, t_Step>>(in8, out8, countLog); break;
            case 10: morph_batch<t_vector_extension, uncompr_f, t_src_l<10, t_Step>>(in8, out8, countLog); break;
            case 11: morph_batch<t_vector_extension, uncompr_f, t_src_l<11, t_Step>>(in8, out8, countLog); break;
            case 12: morph_batch<t_vector_extension, uncompr_f, t_src_l<12, t_Step>>(in8, out8, countLog); break;
            case 13: morph_batch<t_vector_extension, uncompr_f, t_src_l<13, t_Step>>(in8, out8, countLog); break;
            case 14: morph_batch<t_vector_extension, uncompr_f, t_src_l<14, t_Step>>(in8, out8, countLog); break;
            case 15: morph_batch<t_vector_extension, uncompr_f, t_src_l<15, t_Step>>(in8, out8, countLog); break;
            case 16: morph_batch<t_vector_extension, uncompr_f, t_src_l<16, t_Step>>(in8, out8, countLog); break;
            case 17: morph_batch<t_vector_extension, uncompr_f, t_src_l<17, t_Step>>(in8, out8, countLog); break;
            case 18: morph_batch<t_vector_extension, uncompr_f, t_src_l<18, t_Step>>(in8, out8, countLog); break;
            case 19: morph_batch<t_vector_extension, uncompr_f, t_src_l<19, t_Step>>(in8, out8, countLog); break;
            case 20: morph_batch<t_vector_extension, uncompr_f, t_src_l<20, t_Step>>(in8, out8, countLog); break;
            case 21: morph_batch<t_vector_extension, uncompr_f, t_src_l<21, t_Step>>(in8, out8, countLog); break;
            case 22: morph_batch<t_vector_extension, uncompr_f, t_src_l<22, t_Step>>(in8, out8, countLog); break;
            case 23: morph_batch<t_vector_extension, uncompr_f, t_src_l<23, t_Step>>(in8, out8, countLog); break;
            case 24: morph_batch<t_vector_extension, uncompr_f, t_src_l<24, t_Step>>(in8, out8, countLog); break;
            case 25: morph_batch<t_vector_extension, uncompr_f, t_src_l<25, t_Step>>(in8, out8, countLog); break;
            case 26: morph_batch<t_vector_extension, uncompr_f, t_src_l<26, t_Step>>(in8, out8, countLog); break;
#ifndef VBP_LIMIT_ROUTINES_FOR_SSB_SF1
            case 27: morph_batch<t_vector_extension, uncompr_f, t_src_l<27, t_Step>>(in8, out8, countLog); break;
            case 28: morph_batch<t_vector_extension, uncompr_f, t_src_l<28, t_Step>>(in8, out8, countLog); break;
            case 29: morph_batch<t_vector_extension, uncompr_f, t_src_l<29, t_Step>>(in8, out8, countLog); break;
            case 30: morph_batch<t_vector_extension, uncompr_f, t_src_l<30, t_Step>>(in8, out8, countLog); break;
            case 31: morph_batch<t_vector_extension, uncompr_f, t_src_l<31, t_Step>>(in8, out8, countLog); break;
            case 32: morph_batch<t_vector_extension, uncompr_f, t_src_l<32, t_Step>>(in8, out8, countLog); break;
            case 33: morph_batch<t_vector_extension, uncompr_f, t_src_l<33, t_Step>>(in8, out8, countLog); break;
            case 34: morph_batch<t_vector_extension, uncompr_f, t_src_l<34, t_Step>>(in8, out8, countLog); break;
            case 35: morph_batch<t_vector_extension, uncompr_f, t_src_l<35, t_Step>>(in8, out8, countLog); break;
            case 36: morph_batch<t_vector_extension, uncompr_f, t_src_l<36, t_Step>>(in8, out8, countLog); break;
            case 37: morph_batch<t_vector_extension, uncompr_f, t_src_l<37, t_Step>>(in8, out8, countLog); break;
            case 38: morph_batch<t_vector_extension, uncompr_f, t_src_l<38, t_Step>>(in8, out8, countLog); break;
            case 39: morph_batch<t_vector_extension, uncompr_f, t_src_l<39, t_Step>>(in8, out8, countLog); break;
            case 40: morph_batch<t_vector_extension, uncompr_f, t_src_l<40, t_Step>>(in8, out8, countLog); break;
            case 41: morph_batch<t_vector_extension, uncompr_f, t_src_l<41, t_Step>>(in8, out8, countLog); break;
            case 42: morph_batch<t_vector_extension, uncompr_f, t_src_l<42, t_Step>>(in8, out8, countLog); break;
            case 43: morph_batch<t_vector_extension, uncompr_f, t_src_l<43, t_Step>>(in8, out8, countLog); break;
            case 44: morph_batch<t_vector_extension, uncompr_f, t_src_l<44, t_Step>>(in8, out8, countLog); break;
            case 45: morph_batch<t_vector_extension, uncompr_f, t_src_l<45, t_Step>>(in8, out8, countLog); break;
            case 46: morph_batch<t_vector_extension, uncompr_f, t_src_l<46, t_Step>>(in8, out8, countLog); break;
            case 47: morph_batch<t_vector_extension, uncompr_f, t_src_l<47, t_Step>>(in8, out8, countLog); break;
            case 48: morph_batch<t_vector_extension, uncompr_f, t_src_l<48, t_Step>>(in8, out8, countLog); break;
            case 49: morph_batch<t_vector_extension, uncompr_f, t_src_l<49, t_Step>>(in8, out8, countLog); break;
            case 50: morph_batch<t_vector_extension, uncompr_f, t_src_l<50, t_Step>>(in8, out8, countLog); break;
            case 51: morph_batch<t_vector_extension, uncompr_f, t_src_l<51, t_Step>>(in8, out8, countLog); break;
            case 52: morph_batch<t_vector_extension, uncompr_f, t_src_l<52, t_Step>>(in8, out8, countLog); break;
            case 53: morph_batch<t_vector_extension, uncompr_f, t_src_l<53, t_Step>>(in8, out8, countLog); break;
            case 54: morph_batch<t_vector_extension, uncompr_f, t_src_l<54, t_Step>>(in8, out8, countLog); break;
            case 55: morph_batch<t_vector_extension, uncompr_f, t_src_l<55, t_Step>>(in8, out8, countLog); break;
            case 56: morph_batch<t_vector_extension, uncompr_f, t_src_l<56, t_Step>>(in8, out8, countLog); break;
            case 57: morph_batch<t_vector_extension, uncompr_f, t_src_l<57, t_Step>>(in8, out8, countLog); break;
            case 58: morph_batch<t_vector_extension, uncompr_f, t_src_l<58, t_Step>>(in8, out8, countLog); break;
            case 59: morph_batch<t_vector_extension, uncompr_f, t_src_l<59, t_Step>>(in8, out8, countLog); break;
            case 60: morph_batch<t_vector_extension, uncompr_f, t_src_l<60, t_Step>>(in8, out8, countLog); break;
            case 61: morph_batch<t_vector_extension, uncompr_f, t_src_l<61, t_Step>>(in8, out8, countLog); break;
            case 62: morph_batch<t_vector_extension, uncompr_f, t_src_l<62, t_Step>>(in8, out8, countLog); break;
            case 63: morph_batch<t_vector_extension, uncompr_f, t_src_l<63, t_Step>>(in8, out8, countLog); break;
            case 64: morph_batch<t_vector_extension, uncompr_f, t_src_l<64, t_Step>>(in8, out8, countLog); break;
#endif
            case VBP_BW_NOBLOCK: /* do nothing */ break;
#ifdef VBP_ROUTINE_SWITCH_CHECK_BITWIDTH
            default: throw std::runtime_error(
                    "unpack_switch: unsupported bit width: " +
                    std::to_string(bitwidth)
            );
#endif
        }
    }
    
    // ------------------------------------------------------------------------
    // Decompression and processing
    // ------------------------------------------------------------------------

    template<
            class t_vector_extension,
            template<unsigned, unsigned> class t_src_l,
            unsigned t_Step,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
#ifdef VBP_FORCE_INLINE_UNPACK_AND_PROCESS_SWITCH
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
#endif
    void decompress_and_process_batch_switch(
            unsigned bitwidth,
            const uint8_t * & in8,
            size_t countInLog,
            typename t_op_vector<
                    t_vector_extension, t_extra_args ...
            >::state_t & opState
    ) {
        using t_ve = t_vector_extension;
        
        switch(bitwidth) {
            // Generated with Python:
            // for bw in range(1, 64+1):
            //   print("case {: >2}: decompress_and_process_batch<t_ve, t_src_l<{: >2}, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;".format(bw, bw))
            case  1: decompress_and_process_batch<t_ve, t_src_l< 1, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case  2: decompress_and_process_batch<t_ve, t_src_l< 2, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case  3: decompress_and_process_batch<t_ve, t_src_l< 3, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case  4: decompress_and_process_batch<t_ve, t_src_l< 4, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case  5: decompress_and_process_batch<t_ve, t_src_l< 5, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case  6: decompress_and_process_batch<t_ve, t_src_l< 6, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case  7: decompress_and_process_batch<t_ve, t_src_l< 7, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case  8: decompress_and_process_batch<t_ve, t_src_l< 8, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case  9: decompress_and_process_batch<t_ve, t_src_l< 9, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 10: decompress_and_process_batch<t_ve, t_src_l<10, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 11: decompress_and_process_batch<t_ve, t_src_l<11, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 12: decompress_and_process_batch<t_ve, t_src_l<12, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 13: decompress_and_process_batch<t_ve, t_src_l<13, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 14: decompress_and_process_batch<t_ve, t_src_l<14, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 15: decompress_and_process_batch<t_ve, t_src_l<15, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 16: decompress_and_process_batch<t_ve, t_src_l<16, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 17: decompress_and_process_batch<t_ve, t_src_l<17, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 18: decompress_and_process_batch<t_ve, t_src_l<18, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 19: decompress_and_process_batch<t_ve, t_src_l<19, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 20: decompress_and_process_batch<t_ve, t_src_l<20, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 21: decompress_and_process_batch<t_ve, t_src_l<21, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 22: decompress_and_process_batch<t_ve, t_src_l<22, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 23: decompress_and_process_batch<t_ve, t_src_l<23, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 24: decompress_and_process_batch<t_ve, t_src_l<24, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 25: decompress_and_process_batch<t_ve, t_src_l<25, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 26: decompress_and_process_batch<t_ve, t_src_l<26, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
#ifndef VBP_LIMIT_ROUTINES_FOR_SSB_SF1
            case 27: decompress_and_process_batch<t_ve, t_src_l<27, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 28: decompress_and_process_batch<t_ve, t_src_l<28, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 29: decompress_and_process_batch<t_ve, t_src_l<29, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 30: decompress_and_process_batch<t_ve, t_src_l<30, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 31: decompress_and_process_batch<t_ve, t_src_l<31, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 32: decompress_and_process_batch<t_ve, t_src_l<32, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 33: decompress_and_process_batch<t_ve, t_src_l<33, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 34: decompress_and_process_batch<t_ve, t_src_l<34, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 35: decompress_and_process_batch<t_ve, t_src_l<35, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 36: decompress_and_process_batch<t_ve, t_src_l<36, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 37: decompress_and_process_batch<t_ve, t_src_l<37, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 38: decompress_and_process_batch<t_ve, t_src_l<38, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 39: decompress_and_process_batch<t_ve, t_src_l<39, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 40: decompress_and_process_batch<t_ve, t_src_l<40, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 41: decompress_and_process_batch<t_ve, t_src_l<41, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 42: decompress_and_process_batch<t_ve, t_src_l<42, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 43: decompress_and_process_batch<t_ve, t_src_l<43, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 44: decompress_and_process_batch<t_ve, t_src_l<44, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 45: decompress_and_process_batch<t_ve, t_src_l<45, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 46: decompress_and_process_batch<t_ve, t_src_l<46, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 47: decompress_and_process_batch<t_ve, t_src_l<47, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 48: decompress_and_process_batch<t_ve, t_src_l<48, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 49: decompress_and_process_batch<t_ve, t_src_l<49, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 50: decompress_and_process_batch<t_ve, t_src_l<50, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 51: decompress_and_process_batch<t_ve, t_src_l<51, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 52: decompress_and_process_batch<t_ve, t_src_l<52, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 53: decompress_and_process_batch<t_ve, t_src_l<53, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 54: decompress_and_process_batch<t_ve, t_src_l<54, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 55: decompress_and_process_batch<t_ve, t_src_l<55, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 56: decompress_and_process_batch<t_ve, t_src_l<56, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 57: decompress_and_process_batch<t_ve, t_src_l<57, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 58: decompress_and_process_batch<t_ve, t_src_l<58, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 59: decompress_and_process_batch<t_ve, t_src_l<59, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 60: decompress_and_process_batch<t_ve, t_src_l<60, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 61: decompress_and_process_batch<t_ve, t_src_l<61, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 62: decompress_and_process_batch<t_ve, t_src_l<62, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 63: decompress_and_process_batch<t_ve, t_src_l<63, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
            case 64: decompress_and_process_batch<t_ve, t_src_l<64, t_Step>, t_op_vector, t_extra_args ...>::apply(in8, countInLog, opState); break;
#endif
            case VBP_BW_NOBLOCK: /* do nothing */ break;
#ifdef VBP_ROUTINE_SWITCH_CHECK_BITWIDTH
            default: throw std::runtime_error(
                    "decompress_and_process_batch_switch: unsupported bit width: " +
                    std::to_string(bitwidth)
            );
#endif
        }
    }
    
}    

#endif //MORPHSTORE_CORE_MORPHING_VBP_COMMONS_H