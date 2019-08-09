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
 * @file vbp.h
 * @brief The vertical bit-packed layout.
 * 
 * In this particular variant, one packed code word can span across two memory
 * words.
 * 
 * @todo Documentation.
 */

#ifndef MORPHSTORE_CORE_MORPHING_VBP_H
#define MORPHSTORE_CORE_MORPHING_VBP_H

#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/morphing/vbp_commons.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_primitives.h>

#include <limits>

#include <cstdint>

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
    // Layout
    // ************************************************************************
    
    template<unsigned t_Bw, unsigned t_Step>
    struct vbp_l : public layout {
        static_assert(
                (1 <= t_Bw) && (t_Bw <= std::numeric_limits<uint64_t>::digits),
                "vbp_l: template parameter t_Bw must satisfy 1 <= t_Bw <= 64"
        );
        static_assert(
                t_Step > 0,
                "vbp_l: template parameter t_Step must be greater than 0"
        );

        // Assumes that the provided number is a multiple of m_BlockSize.
        static size_t get_size_max_byte(size_t p_CountValues) {
            return p_CountValues * t_Bw / bitsPerByte;
        }
        
        static const size_t m_BlockSize = t_Step * sizeof(uint64_t) * bitsPerByte;
        
        static constexpr unsigned minimum_cycle_len() {
            unsigned cycleLen = std::numeric_limits<uint64_t>::digits;
            unsigned bw = t_Bw;
            while(!(bw & 1)) {
                bw >>= 1;
                cycleLen >>= 1;
            }
            return cycleLen;
        }
    };
    
    
    
    // ************************************************************************
    // Morph-operators (batch-level)
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    
    template<class t_vector_extension, unsigned t_Bw>
    class morph_batch_t<
            t_vector_extension,
            vbp_l<t_Bw, t_vector_extension::vector_helper_t::element_count::value>,
            uncompr_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        using dst_l = vbp_l<t_Bw, t_vector_extension::vector_helper_t::element_count::value>;
        
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
                tmp = vectorlib::set1<t_ve, vector_base_t_granularity::value>(0);
            }
        };
        
        template<unsigned t_CycleLen, unsigned t_PosInCycle>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void pack_block(state_t & s) {
            using namespace vectorlib;
            if(t_CycleLen > 1) {
                pack_block<t_CycleLen / 2, t_PosInCycle                 >(s);
                pack_block<t_CycleLen / 2, t_PosInCycle + t_CycleLen / 2>(s);
            }
            else {
                const vector_t tmp2 = load<t_ve, iov::ALIGNED, vector_size_bit::value>(s.inBase);
                s.inBase += vector_element_count::value;
                s.tmp = bitwise_or<t_ve>(s.tmp, shift_left<t_ve>::apply(tmp2, s.bitpos));
                s.bitpos += t_Bw;
                if(((t_PosInCycle + 1) * t_Bw) % countBits == 0) {
                    store<t_ve, iov::ALIGNED, vector_size_bit::value>(s.outBase, s.tmp);
                    s.outBase += vector_element_count::value;
                    s.tmp = set1<t_ve, vector_base_t_granularity::value>(0);
                    s.bitpos = 0;
                }
                else if(t_PosInCycle * t_Bw / countBits < ((t_PosInCycle + 1) * t_Bw - 1) / countBits) {
                    store<t_ve, iov::ALIGNED, vector_size_bit::value>(s.outBase, s.tmp);
                    s.outBase += vector_element_count::value;
                    s.tmp = shift_right<t_ve>::apply(tmp2, t_Bw - s.bitpos + countBits);
                    s.bitpos -= countBits;
                }
            }
        }
        
    public:
#ifdef VBP_FORCE_INLINE_PACK
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
#endif
        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countInLog
        ) {
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            base_t * outBase = reinterpret_cast<base_t *>(out8);
            state_t s(inBase, outBase);
            
#ifdef VBP_USE_MIN_CYCLE_LEN
            const size_t cycleLenVec = dst_l::minimum_cycle_len();
            const size_t cycleLenBase = cycleLenVec * vector_element_count::value;
            const size_t cycleCount = countInLog / cycleLenBase;
            for(size_t i = 0; i < cycleCount; i++)
                pack_block<cycleLenVec, 0>(s);
#else
            const size_t blockSize = vector_size_bit::value; // @todo use it from vbp_l?
            for(size_t i = 0; i < countInLog; i += blockSize)
                pack_block<countBits, 0>(s);
#endif
            
            in8 = reinterpret_cast<const uint8_t *>(s.inBase);
            out8 = reinterpret_cast<uint8_t *>(s.outBase);
        }
    };
    
    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    
    template<class t_vector_extension, unsigned t_Bw>
    class morph_batch_t<
            t_vector_extension,
            uncompr_f,
            vbp_l<t_Bw, t_vector_extension::vector_helper_t::element_count::value>
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        using src_l = vbp_l<t_Bw, t_vector_extension::vector_helper_t::element_count::value>;
        
        static const size_t countBits = std::numeric_limits<base_t>::digits;
        // @todo It would be nice to initialize this in-class. However, the
        // compiler complains because set1 is not constexpr, even when it is
        // defined so.
        static const vector_t mask; // = vectorlib::set1<t_ve, vector_base_t_granularity::value>(bitwidth_max<base_t>(t_bw));
        
        struct state_t {
            const base_t * inBase;
            base_t * outBase;
            vector_t nextOut;
            unsigned bitpos;
            vector_t tmp;
            
            state_t(const base_t * p_InBase, base_t * p_OutBase) {
                inBase = p_InBase;
                outBase = p_OutBase;
                nextOut = vectorlib::set1<t_ve, vector_base_t_granularity::value>(0);
                // @todo Maybe we don't need this.
                bitpos = 0;
                tmp = vectorlib::set1<t_ve, vector_base_t_granularity::value>(0);
            }
        };
        
        template<unsigned t_CycleLen, unsigned t_PosInCycle>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void unpack_block(state_t & s) {
            using namespace vectorlib;
            
            if(t_CycleLen > 1) {
                unpack_block<t_CycleLen / 2, t_PosInCycle                 >(s);
                unpack_block<t_CycleLen / 2, t_PosInCycle + t_CycleLen / 2>(s);
            }
            else {
                if((t_PosInCycle * t_Bw) % countBits == 0) {
                    s.tmp = load<t_ve, iov::ALIGNED, vector_size_bit::value>(s.inBase);
                    s.inBase += vector_element_count::value;
                    s.nextOut = bitwise_and<t_ve>(mask, s.tmp);
                    s.bitpos = t_Bw;
                }
                else if(t_PosInCycle * t_Bw / countBits < ((t_PosInCycle + 1) * t_Bw - 1) / countBits) {
                    s.tmp = load<t_ve, iov::ALIGNED, vector_size_bit::value>(s.inBase);
                    s.inBase += vector_element_count::value;
                    s.nextOut = bitwise_and<t_ve>(mask, bitwise_or<t_ve>(shift_left<t_ve>::apply(s.tmp, countBits - s.bitpos + t_Bw), s.nextOut));
                    s.bitpos = s.bitpos - countBits;
                }
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(s.outBase, s.nextOut);
                s.outBase += vector_element_count::value;
                s.nextOut = bitwise_and<t_ve>(mask, shift_right<t_ve>::apply(s.tmp, s.bitpos));
                s.bitpos += t_Bw;
            }
        }
        
    public:
#ifdef VBP_FORCE_INLINE_UNPACK
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
#endif
        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countLog
        ) {
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            base_t * outBase = reinterpret_cast<base_t *>(out8);
            state_t s(inBase, outBase);
#ifdef VBP_USE_MIN_CYCLE_LEN
            const size_t cycleLenVec = src_l::minimum_cycle_len();
            const size_t cycleLenBase = cycleLenVec * vector_element_count::value;
            const size_t cycleCount = countLog / cycleLenBase;
            for(size_t i = 0; i < cycleCount; i++)
                unpack_block<cycleLenVec, 0>(s);
#else
            const size_t blockSize = vector_size_bit::value;
            for(size_t i = 0; i < countLog; i += blockSize)
                unpack_block<countBits, 0>(s);
#endif
            
            in8 = reinterpret_cast<const uint8_t *>(s.inBase);
            out8 = reinterpret_cast<uint8_t *>(s.outBase);
        }
    };
    
    template<class t_vector_extension, unsigned t_Bw>
    const typename t_vector_extension::vector_t morph_batch_t<
            t_vector_extension,
            uncompr_f,
            vbp_l<t_Bw, t_vector_extension::vector_helper_t::element_count::value>
    >::mask = vectorlib::set1<
            t_vector_extension,
            t_vector_extension::vector_helper_t::granularity::value
    >(
            bitwidth_max<typename t_vector_extension::base_t>(t_Bw)
    );
    
    
    
    // ************************************************************************
    // Interfaces for accessing compressed data
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Sequential read
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            unsigned t_bw,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    class decompress_and_process_batch<
            t_vector_extension,
            vbp_l<t_bw, t_vector_extension::vector_helper_t::element_count::value>,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        using src_l = vbp_l<t_bw, t_vector_extension::vector_helper_t::element_count::value>;
        
        static const size_t countBits = std::numeric_limits<base_t>::digits;
        // @todo It would be nice to initialize this in-class. However, the
        // compiler complains because set1 is not constexpr, even when it is
        // defined so.
        static const vector_t mask; // = vectorlib::set1<t_ve, vector_base_t_granularity::value>(bitwidth_max<base_t>(t_bw));
        
        struct state_t {
            const base_t * inBase;
            vector_t nextOut;
            unsigned bitpos;
            vector_t tmp;
            
            state_t(const base_t * p_InBase) {
                inBase = p_InBase;
                nextOut = vectorlib::set1<t_ve, vector_base_t_granularity::value>(0);
                // @todo Maybe we don't need this.
                bitpos = 0;
                tmp = vectorlib::set1<t_ve, vector_base_t_granularity::value>(0);
            }
        };
        
        template<unsigned t_CycleLen, unsigned t_PosInCycle>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static void unpack_and_process_block(
                state_t & s,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & opState
        ) {
            using namespace vectorlib;
            
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
#ifdef VBP_FORCE_INLINE_UNPACK_AND_PROCESS
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
#endif
        static void apply(
                const uint8_t * & in8,
                size_t countIn8,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & opState
        ) {
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            const base_t * const endInBase = inBase + convert_size<uint8_t, base_t>(countIn8);
            state_t s(inBase);
#ifdef VBP_USE_MIN_CYCLE_LEN
            const unsigned cycleLen = src_l::minimum_cycle_len();
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
    const typename t_vector_extension::vector_t decompress_and_process_batch<
            t_vector_extension,
            vbp_l<t_bw, t_vector_extension::vector_helper_t::element_count::value>,
            t_op_vector,
            t_extra_args ...
    >::mask = vectorlib::set1<
            t_vector_extension,
            t_vector_extension::vector_helper_t::granularity::value
    >(
            bitwidth_max<typename t_vector_extension::base_t>(t_bw)
    );
    
    // ------------------------------------------------------------------------
    // Random read
    // ------------------------------------------------------------------------

    template<class t_vector_extension, unsigned t_Bw, unsigned t_Step>
    class random_read_access<t_vector_extension, vbp_l<t_Bw, t_Step> > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        const base_t * const m_Data;
        
        static const unsigned m_Shift4DivStep = (t_Step == 1) ? 0 : effective_bitwidth(t_Step - 1);
        static const unsigned m_ShiftDivBaseBits = effective_bitwidth(std::numeric_limits<base_t>::digits - 1);
        
        // @todo It would be nice if these were const, but this is currently
        // not possible, because vectorlib::set1 cannot be used as a constant
        // expression.
        static vector_t m_BaseBitsVec;
        static vector_t m_Mask4ModStep;
        static vector_t m_MaskModBaseBits;
        static vector_t m_BwVec;
        static vector_t m_StepVec;
        static vector_t m_MaskDecompr;
        
    public:
        random_read_access(const base_t * p_Data) : m_Data(p_Data) {
            //
        }

        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        vector_t get(const vector_t & p_Positions) {
            using namespace vectorlib;
            
            const vector_t elemIdxInVec = bitwise_and<t_ve>(
                    p_Positions, m_Mask4ModStep
            );
            const vector_t bitPosInBuf = mul<t_ve>::apply(
                    shift_right<t_ve>::apply(p_Positions, m_Shift4DivStep),
                    m_BwVec
            );
            const vector_t vecIdxs = shift_right<t_ve>::apply(
                    bitPosInBuf, m_ShiftDivBaseBits
            );
            const vector_t baseIdxs = add<t_ve>::apply(
                    shift_left<t_ve>::apply(vecIdxs, m_Shift4DivStep),
                    elemIdxInVec
            );
            
            vector_t res = vectorlib::gather<
                    t_ve, iov::UNALIGNED, vector_base_t_granularity::value
            >(m_Data, baseIdxs);
            
            const vector_t bitPosInElem = bitwise_and<t_ve>(
                    bitPosInBuf, m_MaskModBaseBits
            );
            
            res = shift_right_individual<t_ve>::apply(res, bitPosInElem);
            
            const vector_t nextBaseIdxs = add<t_ve>::apply(baseIdxs, m_StepVec);
            
            const vector_t shiftUpper = sub<t_ve>::apply(
                    m_BaseBitsVec, bitPosInElem
            );
            const vector_t gatherUpper = vectorlib::gather<
                    t_ve,
                    iov::UNALIGNED,
                    vector_base_t_granularity::value
            >(m_Data, nextBaseIdxs);
            const vector_t resUpper = shift_left_individual<t_ve>::apply(
                    gatherUpper, shiftUpper
            );
            res = bitwise_and<t_ve>(
                    m_MaskDecompr, bitwise_or<t_ve>(resUpper, res)
            );
            
            return res;
        }
    };

    template<class t_ve, unsigned t_Bw, unsigned t_Step>
    typename t_ve::vector_t random_read_access<
            t_ve, vbp_l<t_Bw, t_Step>
    >::m_BaseBitsVec = vectorlib::set1<
            t_ve, t_ve::vector_helper_t::granularity::value
    >(std::numeric_limits<typename t_ve::base_t>::digits);

    template<class t_ve, unsigned t_Bw, unsigned t_Step>
    typename t_ve::vector_t random_read_access<
            t_ve, vbp_l<t_Bw, t_Step>
    >::m_Mask4ModStep = vectorlib::set1<
            t_ve, t_ve::vector_helper_t::granularity::value
    >(t_Step - 1);

    template<class t_ve, unsigned t_Bw, unsigned t_Step>
    typename t_ve::vector_t random_read_access<
            t_ve, vbp_l<t_Bw, t_Step>
    >::m_MaskModBaseBits = vectorlib::set1<
            t_ve, t_ve::vector_helper_t::granularity::value
    >(std::numeric_limits<typename t_ve::base_t>::digits - 1);

    template<class t_ve, unsigned t_Bw, unsigned t_Step>
    typename t_ve::vector_t random_read_access<
            t_ve, vbp_l<t_Bw, t_Step>
    >::m_BwVec = vectorlib::set1<
            t_ve, t_ve::vector_helper_t::granularity::value
    >(t_Bw);

    template<class t_ve, unsigned t_Bw, unsigned t_Step>
    typename t_ve::vector_t random_read_access<
            t_ve, vbp_l<t_Bw, t_Step>
    >::m_StepVec = vectorlib::set1<
            t_ve, t_ve::vector_helper_t::granularity::value
    >(t_Step);
    
    template<class t_ve, unsigned t_Bw, unsigned t_Step>
    typename t_ve::vector_t random_read_access<
            t_ve, vbp_l<t_Bw, t_Step>
    >::m_MaskDecompr = vectorlib::set1<
            t_ve, t_ve::vector_helper_t::granularity::value
    >(bitwidth_max<typename t_ve::base_t>(t_Bw));
}

#endif //MORPHSTORE_CORE_MORPHING_VBP_H