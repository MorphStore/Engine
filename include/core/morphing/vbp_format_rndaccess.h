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
 * @file vbp_format_rndaccess.h
 * @brief Format definition and random access for the vertical bit-packed
 * layout.
 * 
 * In this particular variant, one packed code word can span across two memory
 * words.
 * 
 * @todo Documentation.
 */

#ifndef MORPHSTORE_CORE_MORPHING_VBP_FORMAT_RNDACCESS_H
#define MORPHSTORE_CORE_MORPHING_VBP_FORMAT_RNDACCESS_H

#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/morphing/vbp_commons.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_primitives.h>

#include <limits>
#include <type_traits>

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
    
    
    // ------------------------------------------------------------------------
    // Random read
    // ------------------------------------------------------------------------
    
    namespace _random_read_access_variants {
        // @todo Think about the restrictions caused by the precision of the
        // multiplication.

        /**
         * @brief `random_read_access` for `vbp_l` with a step of 1 (horizontal
         * bit packing). Works only if the bit width is a power of two and
         * greater than or equal to 8, i.e., for 8, 16, 32, 64 bit.
         */
        template<class t_vector_extension, unsigned t_Bw>
        class rra_vbp_l_hor_8_16_32_64 {
            static_assert(
                    t_Bw == 8 || t_Bw == 16 || t_Bw == 32 || t_Bw == 64,
                    "rra_vbp_l_hor_8_16_32_64: template parameter t_Bw must be "
                    "8, 16, 32, or 64, otherwise the results would be incorrect"
            );
            
            using t_ve = t_vector_extension;
            IMPORT_VECTOR_BOILER_PLATE(t_ve)
            
            const base_t * const m_Data;
            
            static const vector_t m_MaskDecompr;
            
        public:
            rra_vbp_l_hor_8_16_32_64(const base_t * p_Data) : m_Data(p_Data) {
                //
            }
            
            MSV_CXX_ATTRIBUTE_FORCE_INLINE
            vector_t get(const vector_t & p_Positions) {
                using namespace vectorlib;
                
                // For 64-bit we do not even need the mask, but for 64-bit, we
                // should rather use uncompressed data anyway.
                return bitwise_and<t_ve>(
                        gather<
                                t_ve,
                                vector_base_t_granularity::value,
                                t_Bw / bitsPerByte
                        >(m_Data, p_Positions),
                        m_MaskDecompr
                );
            }
        };

        template<class t_ve, unsigned t_Bw>
        const typename t_ve::vector_t rra_vbp_l_hor_8_16_32_64<
                t_ve, t_Bw
        >::m_MaskDecompr = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(bitwidth_max<uint64_t>(t_Bw));
        
        /**
         * @brief `random_read_access` for `vbp_l` with a step of 1 (horizontal
         * bit packing). Works only if the bit width is a power of two and less
         * than or equal to 8, i.e., for 1, 2, 4, 8 bit.
         */
        template<class t_vector_extension, unsigned t_Bw>
        class rra_vbp_l_hor_1_2_4_8 {
            static_assert(
                    t_Bw == 1 || t_Bw == 2 || t_Bw == 4 || t_Bw == 8,
                    "rra_vbp_l_hor_1_2_4_8: template parameter t_Bw must be "
                    "1, 2, 4, or 8, otherwise the results would be incorrect"
            );
            
            using t_ve = t_vector_extension;
            IMPORT_VECTOR_BOILER_PLATE(t_ve)
            
            const base_t * const m_Data;
            
            static const unsigned m_Shift4DivCodesPerByte = shift_for_muldiv(
                    bitsPerByte / t_Bw
            );
            static const unsigned m_Shift4MulBw = shift_for_muldiv(t_Bw);
            
            static const vector_t m_Mask4ModBitsPerByte;
            static const vector_t m_MaskDecompr;
            
        public:
            rra_vbp_l_hor_1_2_4_8(const base_t * p_Data) : m_Data(p_Data) {
                //
            }
            
            MSV_CXX_ATTRIBUTE_FORCE_INLINE
            vector_t get(const vector_t & p_Positions) {
                using namespace vectorlib;
                
                return bitwise_and<t_ve>(
                        shift_right_individual<t_ve>::apply(
                                gather<
                                        t_ve,
                                        vector_base_t_granularity::value,
                                        1
                                >(
                                        m_Data,
                                        shift_right<t_ve>::apply(
                                                p_Positions,
                                                m_Shift4DivCodesPerByte
                                        )
                                ),
                                bitwise_and<t_ve>(
                                        shift_left<t_ve>::apply(
                                                p_Positions, m_Shift4MulBw
                                        ),
                                        m_Mask4ModBitsPerByte
                                )
                        ),
                        m_MaskDecompr
                );
            }
        };

        template<class t_ve, unsigned t_Bw>
        const typename t_ve::vector_t rra_vbp_l_hor_1_2_4_8<
                t_ve, t_Bw
        >::m_Mask4ModBitsPerByte = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(mask_for_mod(bitsPerByte));

        template<class t_ve, unsigned t_Bw>
        const typename t_ve::vector_t rra_vbp_l_hor_1_2_4_8<
                t_ve, t_Bw
        >::m_MaskDecompr = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(bitwidth_max<uint64_t>(t_Bw));
        
        /**
         * @brief `random_read_access` for `vbp_l` with a step of 1 (horizontal
         * bit packing). Works only if the bit width is a multiple of 8, i.e.,
         * for 8, 16, 24, 32, 40, 48, 56, 64 bit.
         */
        template<class t_vector_extension, unsigned t_Bw>
        class rra_vbp_l_hor_full_byte {
            using t_ve = t_vector_extension;
            IMPORT_VECTOR_BOILER_PLATE(t_ve)
            
            const base_t * const m_Data;
            
            static const vector_t m_ByteWidth;
            static const vector_t m_MaskDecompr;
            
        public:
            rra_vbp_l_hor_full_byte(const base_t * p_Data) : m_Data(p_Data) {
                //
            }
            
            MSV_CXX_ATTRIBUTE_FORCE_INLINE
            vector_t get(const vector_t & p_Positions) {
                using namespace vectorlib;
                
                // For 64-bit we do not even need the mask, but for 64-bit, we
                // should rather use uncompressed data anyway.
                return bitwise_and<t_ve>(
                        gather<
                                t_ve,
                                vector_base_t_granularity::value,
                                1
                        >(
                                m_Data,
                                mul<t_ve>::apply(p_Positions, m_ByteWidth)
                        ),
                        m_MaskDecompr
                );
            }
        };

        template<class t_ve, unsigned t_Bw>
        const typename t_ve::vector_t rra_vbp_l_hor_full_byte<
                t_ve, t_Bw
        >::m_ByteWidth = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(t_Bw / bitsPerByte);

        template<class t_ve, unsigned t_Bw>
        const typename t_ve::vector_t rra_vbp_l_hor_full_byte<
                t_ve, t_Bw
        >::m_MaskDecompr = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(bitwidth_max<uint64_t>(t_Bw));
        
        /**
         * @brief `random_read_access` for `vbl_l` with a step of 1 (horizontal
         * bit packing). Works for all bit widths except for 59, 61, 62, 63
         * bit.
         */
        template<class t_vector_extension, unsigned t_Bw>
        class rra_vbp_l_hor_all_but_59_61_62_63 {
            static_assert(
                    !( t_Bw == 59 || (t_Bw >= 61 && t_Bw <= 63) ),
                    "rra_vbp_l_hor_all_but_59_61_62_63: template parameter "
                    "t_Bw must not be 59, 61, 62, or 63, otherwise the "
                    "results would be incorrect"
            );
            
            using t_ve = t_vector_extension;
            IMPORT_VECTOR_BOILER_PLATE(t_ve)
            
            const base_t * const m_Data;
            
            static const unsigned m_Shift4DivBitsPerByte = shift_for_muldiv(
                    bitsPerByte
            );
            
            static const vector_t m_Bw;
            static const vector_t m_Mask4ModBitsPerByte;
            static const vector_t m_MaskDecompr;
            
        public:
            rra_vbp_l_hor_all_but_59_61_62_63(const base_t * p_Data)
            : m_Data(p_Data) {
                //
            }
            
            MSV_CXX_ATTRIBUTE_FORCE_INLINE
            vector_t get(const vector_t & p_Positions) {
                using namespace vectorlib;
                
                const vector_t bitPositions = mul<t_ve>::apply(
                        p_Positions, m_Bw
                );
                // For 64-bit we do not even need the mask, but for 64-bit, we
                // should rather use uncompressed data anyway.
                return bitwise_and<t_ve>(
                        shift_right_individual<t_ve>::apply(
                                gather<
                                        t_ve,
                                        vector_base_t_granularity::value,
                                        1
                                >(
                                        m_Data,
                                        shift_right<t_ve>::apply(
                                                bitPositions,
                                                m_Shift4DivBitsPerByte
                                        )
                                ),
                                bitwise_and<t_ve>(
                                        bitPositions,
                                        m_Mask4ModBitsPerByte
                                )
                        ),
                        m_MaskDecompr
                );
            }
        };

        template<class t_ve, unsigned t_Bw>
        const typename t_ve::vector_t rra_vbp_l_hor_all_but_59_61_62_63<
                t_ve, t_Bw
        >::m_Bw = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(t_Bw);

        template<class t_ve, unsigned t_Bw>
        const typename t_ve::vector_t rra_vbp_l_hor_all_but_59_61_62_63<
                t_ve, t_Bw
        >::m_Mask4ModBitsPerByte = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(mask_for_mod(bitsPerByte));

        template<class t_ve, unsigned t_Bw>
        const typename t_ve::vector_t rra_vbp_l_hor_all_but_59_61_62_63<
                t_ve, t_Bw
        >::m_MaskDecompr = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(bitwidth_max<uint64_t>(t_Bw));
        
        /**
         * @brief `random_read_access` for `vbp_l` with any step (vertical and
         * horizontal layout). Works only if the bit width is a power of two,
         * i.e., for 1, 2, 4, 8, 16, 32, 64 bit.
         */
        template<class t_vector_extension, unsigned t_Bw, unsigned t_Step>
        class rra_vbp_l_1_2_4_8_16_32_64 {
            static_assert(
                    is_power_of_two(t_Bw),
                    "rra_vbp_l_1_2_4_8_16_32_64: template parameter t_Bw must "
                    "be a power of two, otherwise the results would be "
                    "incorrect"
            );
            
            using t_ve = t_vector_extension;
            IMPORT_VECTOR_BOILER_PLATE(t_ve)

            const base_t * const m_Data;
     
            static const unsigned m_Shift4DivStep = shift_for_muldiv(t_Step);
            static const unsigned m_Shift4MulBw = shift_for_muldiv(t_Bw);
            static const unsigned m_Shift4DivBaseBits = shift_for_muldiv(
                    std::numeric_limits<base_t>::digits
            );
            static const unsigned m_Shift4MulStep = shift_for_muldiv(t_Step);
            
            static const vector_t m_Mask4ModStep;
            static const vector_t m_Mask4ModBaseBits;
            static const vector_t m_MaskDecompr;

        public:
            rra_vbp_l_1_2_4_8_16_32_64(const base_t * p_Data) : m_Data(p_Data) {
                //
            }
            
            MSV_CXX_ATTRIBUTE_FORCE_INLINE
            vector_t get(const vector_t & p_Positions) {
                using namespace vectorlib;
                
                const vector_t bitPosInStep = shift_left<t_ve>::apply(
                        shift_right<t_ve>::apply(p_Positions, m_Shift4DivStep),
                        m_Shift4MulBw
                );
                return bitwise_and<t_ve>(
                        m_MaskDecompr,
                        shift_right_individual<t_ve>::apply(
                                vectorlib::gather<
                                        t_ve,
                                        vector_base_t_granularity::value,
                                        sizeof(base_t)
                                >(
                                        m_Data,
                                        // Bitwise OR instead of addition would
                                        // be possible, too.
                                        add<t_ve>::apply(
                                                shift_left<t_ve>::apply(
                                                        shift_right<t_ve>::apply(
                                                                bitPosInStep,
                                                                m_Shift4DivBaseBits
                                                        ),
                                                        m_Shift4MulStep
                                                ),
                                                bitwise_and<t_ve>(
                                                        p_Positions,
                                                        m_Mask4ModStep
                                                )
                                        )
                                ),
                                bitwise_and<t_ve>(
                                        bitPosInStep, m_Mask4ModBaseBits
                                )
                        )
                );
            }
        };

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_l_1_2_4_8_16_32_64<
                t_ve, t_Bw, t_Step
        >::m_Mask4ModStep = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(mask_for_mod(t_Step));

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_l_1_2_4_8_16_32_64<
                t_ve, t_Bw, t_Step
        >::m_Mask4ModBaseBits = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(mask_for_mod(std::numeric_limits<typename t_ve::base_t>::digits));

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_l_1_2_4_8_16_32_64<
                t_ve, t_Bw, t_Step
        >::m_MaskDecompr = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(bitwidth_max<typename t_ve::base_t>(t_Bw));

        /**
         * @brief `random_read_access` for `vbp_l`. Works for all combinations
         * of the step and the bit width.
         */
        template<class t_vector_extension, unsigned t_Bw, unsigned t_Step>
        class rra_vbp_l_general {
            using t_ve = t_vector_extension;
            IMPORT_VECTOR_BOILER_PLATE(t_ve)

            const base_t * const m_Data;

            static const unsigned m_Shift4DivStep = shift_for_muldiv(t_Step);
            static const unsigned m_Shift4DivBaseBits = shift_for_muldiv(
                    std::numeric_limits<base_t>::digits
            );
            static const unsigned m_Shift4MulStep = shift_for_muldiv(t_Step);

            static const vector_t m_BaseBitsVec;
            static const vector_t m_Mask4ModStep;
            static const vector_t m_Mask4ModBaseBits;
            static const vector_t m_BwVec;
            static const vector_t m_StepVec;
            static const vector_t m_MaskDecompr;

        public:
            rra_vbp_l_general(const base_t * p_Data) : m_Data(p_Data) {
                //
            }

            MSV_CXX_ATTRIBUTE_FORCE_INLINE
            vector_t get(const vector_t & p_Positions) {
                using namespace vectorlib;

                const vector_t bitPosInStep = mul<t_ve>::apply(
                        shift_right<t_ve>::apply(p_Positions, m_Shift4DivStep),
                        m_BwVec
                );
                const vector_t bitPosInElem = bitwise_and<t_ve>(
                        bitPosInStep, m_Mask4ModBaseBits
                );
                // Bitwise OR instead of addition would be possible, too.
                const vector_t baseIdxs = add<t_ve>::apply(
                        shift_left<t_ve>::apply(
                                shift_right<t_ve>::apply(
                                        bitPosInStep, m_Shift4DivBaseBits
                                ),
                                m_Shift4MulStep
                        ),
                        bitwise_and<t_ve>(p_Positions, m_Mask4ModStep)
                );
                return bitwise_and<t_ve>(
                        m_MaskDecompr,
                        bitwise_or<t_ve>(
                                shift_right_individual<t_ve>::apply(
                                        vectorlib::gather<
                                                t_ve,
                                                vector_base_t_granularity::value,
                                                sizeof(base_t)
                                        >(m_Data, baseIdxs),
                                        bitPosInElem
                                ),
                                shift_left_individual<t_ve>::apply(
                                        vectorlib::gather<
                                                t_ve,
                                                vector_base_t_granularity::value,
                                                sizeof(base_t)
                                        >(
                                                m_Data,
                                                add<t_ve>::apply(
                                                        baseIdxs, m_StepVec
                                                )
                                        ),
                                        sub<t_ve>::apply(
                                                m_BaseBitsVec, bitPosInElem
                                        )
                                )
                        )
                );
            }
        };

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_l_general<
                t_ve, t_Bw, t_Step
        >::m_BaseBitsVec = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(std::numeric_limits<typename t_ve::base_t>::digits);

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_l_general<
                t_ve, t_Bw, t_Step
        >::m_Mask4ModStep = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(mask_for_mod(t_Step));

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_l_general<
                t_ve, t_Bw, t_Step
        >::m_Mask4ModBaseBits = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(mask_for_mod(std::numeric_limits<typename t_ve::base_t>::digits));

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_l_general<
                t_ve, t_Bw, t_Step
        >::m_BwVec = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(t_Bw);

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_l_general<
                t_ve, t_Bw, t_Step
        >::m_StepVec = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(t_Step);

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_l_general<
                t_ve, t_Bw, t_Step
        >::m_MaskDecompr = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(bitwidth_max<typename t_ve::base_t>(t_Bw));
    
    }
    
    template<class t_vector_extension, unsigned t_Bw, unsigned t_Step>
    struct random_read_access<t_vector_extension, vbp_l<t_Bw, t_Step> > {
        // Alias to the most efficient implementation we have, depending on the
        // step and the bit width.
        using type = typename std::conditional<
                t_Step == 1,
                // Horizontal layout.
                typename std::conditional<
                        is_power_of_two(t_Bw),
                        // Bit width is a power of two.
                        typename std::conditional<
                                t_Bw < 8,
                                // Bit width is a sub-byte power of two.
                                _random_read_access_variants::rra_vbp_l_hor_1_2_4_8<
                                        t_vector_extension, t_Bw
                                >,
                                // Bit width is a byte power of two.
                                _random_read_access_variants::rra_vbp_l_hor_8_16_32_64<
                                        t_vector_extension, t_Bw
                                >
                        >::type,
                        // Bit width is not a power of two.
                        typename std::conditional<
                                t_Bw % 8 == 0,
                                // Bit width is a full byte width.
                                _random_read_access_variants::rra_vbp_l_hor_full_byte<
                                        t_vector_extension, t_Bw
                                >,
                                // Bit width is not a full byte width.
                                typename std::conditional<
                                        !(t_Bw == 59 || (t_Bw >= 61 && t_Bw <= 63)),
                                        // Bit width is still ok-ish.
                                        _random_read_access_variants::rra_vbp_l_hor_all_but_59_61_62_63<
                                                t_vector_extension, t_Bw
                                        >,
                                        // Bit width is really ugly.
                                        _random_read_access_variants::rra_vbp_l_general<
                                                t_vector_extension, t_Bw, t_Step
                                        >
                                >::type
                        >::type
                >::type,
                // Vertical layout.
                typename std::conditional<
                        is_power_of_two(t_Bw),
                        // Bit width is a power of two.
                        _random_read_access_variants::rra_vbp_l_1_2_4_8_16_32_64<
                                t_vector_extension, t_Bw, t_Step
                        >,
                        // Bit width is not a power of two.
                        _random_read_access_variants::rra_vbp_l_general<
                                t_vector_extension, t_Bw, t_Step
                        >
                >::type
        >::type;
    };
}

#endif //MORPHSTORE_CORE_MORPHING_VBP_FORMAT_RNDACCESS_H