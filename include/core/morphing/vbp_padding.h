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
 * @file vbp_padding.h
 * @brief The vertical bit-packed layout with padding in each 64-bit word.
 * 
 * In this particular variant, one packed code word can never span across two
 * memory words.
 * 
 * @todo Documentation.
 */

#ifndef MORPHSTORE_CORE_MORPHING_VBP_PADDING_H
#define MORPHSTORE_CORE_MORPHING_VBP_PADDING_H

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

namespace morphstore {
    
    // ************************************************************************
    // Layout
    // ************************************************************************
    
    template<unsigned t_Bw, unsigned t_Step>
    struct vbp_padding_l : public layout {
        // @todo Code duplication: Move this into a common base class.
        static_assert(
                (1 <= t_Bw) && (t_Bw <= std::numeric_limits<uint64_t>::digits),
                "vbp_padding_l: template parameter t_Bw must satisfy 1 <= t_Bw <= 64"
        );
        static_assert(
                t_Step > 0,
                "vbp_paddingl: template parameter t_Step must be greater than 0"
        );

        // Assumes that the provided number is a multiple of m_BlockSize.
        static size_t get_size_max_byte(size_t p_CountValues) {
            return p_CountValues / m_BlockSize * t_Step * sizeof(uint64_t) * 10;
        }
        
        // @todo Think about this again.
        static const size_t m_BlockSize =
                std::numeric_limits<uint64_t>::digits / t_Bw * t_Step;
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
            vbp_padding_l<
                    t_Bw,
                    t_vector_extension::vector_helper_t::element_count::value
            >,
            uncompr_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        using dst_l =vbp_padding_l<
                t_Bw, t_vector_extension::vector_helper_t::element_count::value
        >;
        
        // @todo We could use unrolled routines as we did with vbp_l.
        
    public:
#ifdef VBP_FORCE_INLINE_PACK
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
#endif
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countInLog
        ) {
            using namespace vectorlib;
            
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            base_t * outBase = reinterpret_cast<base_t *>(out8);
            
            const size_t blockSizeVec =
                    dst_l::m_BlockSize / vector_element_count::value;
            
            for(size_t i = 0; i < countInLog; i += dst_l::m_BlockSize) {
                vector_t tmp = load<
                        t_ve, iov::ALIGNED, vector_size_bit::value
                >(inBase);
                inBase += vector_element_count::value;
                for(size_t k = 1; k < blockSizeVec; k++) {
                    tmp = bitwise_or<t_ve>(
                            tmp,
                            shift_left<t_ve>::apply(
                                    load<
                                            t_ve,
                                            iov::ALIGNED,
                                            vector_size_bit::value
                                    >(inBase),
                                    k * t_Bw
                            )
                    );
                    inBase += vector_element_count::value;
                }
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        outBase, tmp
                );
                outBase += vector_element_count::value;
            }
            
            in8 = reinterpret_cast<const uint8_t *>(inBase);
            out8 = reinterpret_cast<uint8_t *>(outBase);
        }
    };
    
    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    
    template<class t_vector_extension, unsigned t_Bw>
    class morph_batch_t<
            t_vector_extension,
            uncompr_f,
            vbp_padding_l<
                    t_Bw,
                    t_vector_extension::vector_helper_t::element_count::value
            >
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        using src_l = vbp_padding_l<
                t_Bw, t_vector_extension::vector_helper_t::element_count::value
        >;
        
        static const vector_t mask;
        
        // @todo We could use unrolled routines as we did with vbp_l.
        
    public:
#ifdef VBP_FORCE_INLINE_UNPACK
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
#endif
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
            using namespace vectorlib;
            
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            base_t * outBase = reinterpret_cast<base_t *>(out8);
            
            const size_t blockSizeVec =
                    src_l::m_BlockSize / vector_element_count::value;
            
            for(size_t i = 0; i < countLog; i += src_l::m_BlockSize) {
                const vector_t tmp = load<
                        t_ve, iov::ALIGNED, vector_size_bit::value
                >(inBase);
                inBase += vector_element_count::value;
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        outBase, bitwise_and<t_ve>(mask, tmp)
                );
                outBase += vector_element_count::value;
                for(size_t k = 1; k < blockSizeVec; k++) {
                    store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                            outBase,
                            // @todo The last one does not need to be masked.
                            bitwise_and<t_ve>(
                                    mask,
                                    shift_right<t_ve>::apply(tmp, k * t_Bw)
                            )
                    );
                    outBase += vector_element_count::value;
                }
            }
            
            in8 = reinterpret_cast<const uint8_t *>(inBase);
            out8 = reinterpret_cast<uint8_t *>(outBase);
        }
    };
    
    template<class t_vector_extension, unsigned t_Bw>
    const typename t_vector_extension::vector_t morph_batch_t<
            t_vector_extension,
            uncompr_f,
            vbp_padding_l<
                    t_Bw,
                    t_vector_extension::vector_helper_t::element_count::value
            >
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
            unsigned t_Bw,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    class decompress_and_process_batch<
            t_vector_extension,
            vbp_padding_l<
                    t_Bw,
                    t_vector_extension::vector_helper_t::element_count::value
            >,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        using src_l = vbp_padding_l<
                t_Bw,
                t_vector_extension::vector_helper_t::element_count::value
        >;
        
        static const vector_t mask;
        
        // @todo We could use unrolled routines as we did with vbp_l.
        
    public:
#ifdef VBP_FORCE_INLINE_UNPACK_AND_PROCESS
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
#endif
        static void apply(
                const uint8_t * & in8,
                size_t countIn8,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & opState
        ) {
            using namespace vectorlib;
            
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            const base_t * const endInBase =
                    inBase + convert_size<uint8_t, base_t>(countIn8);
            
            const size_t blockSizeVec =
                    src_l::m_BlockSize / vector_element_count::value;
            
            while(inBase < endInBase) {
                const vector_t tmp = load<
                        t_ve, iov::ALIGNED, vector_size_bit::value
                >(inBase);
                inBase += vector_element_count::value;
                t_op_vector<t_ve, t_extra_args ...>::apply(
                        bitwise_and<t_ve>(mask, tmp), opState
                );
                for(size_t k = 1; k < blockSizeVec; k++) {
                    t_op_vector<t_ve, t_extra_args ...>::apply(
                            // @todo The last one does not need to be masked.
                            bitwise_and<t_ve>(
                                    mask,
                                    shift_right<t_ve>::apply(tmp, k * t_Bw)
                            ),
                            opState
                    );
                }
            }
            
            in8 = reinterpret_cast<const uint8_t *>(inBase);
        }
    };
    
    template<
            class t_vector_extension,
            unsigned t_Bw,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    const typename t_vector_extension::vector_t decompress_and_process_batch<
            t_vector_extension,
            vbp_padding_l<
                    t_Bw,
                    t_vector_extension::vector_helper_t::element_count::value
            >,
            t_op_vector,
            t_extra_args ...
    >::mask = vectorlib::set1<
            t_vector_extension,
            t_vector_extension::vector_helper_t::granularity::value
    >(
            bitwidth_max<typename t_vector_extension::base_t>(t_Bw)
    );
    
    // ------------------------------------------------------------------------
    // Random read
    // ------------------------------------------------------------------------
    
    namespace _random_read_access_variants {

        /**
         * @brief An implementation of the `random_read_access` interface for
         * the layout `vbp_padding_l` which works for all combinations of the
         * bit width and the step, but is quite inefficient.
         */
        template<class t_vector_extension, unsigned t_Bw, unsigned t_Step>
        class rra_vbp_padding_l_general {
            using t_ve = t_vector_extension;
            IMPORT_VECTOR_BOILER_PLATE(t_ve)

            const base_t * const m_Data;

            static const unsigned m_Shift4DivStep = shift_for_muldiv(t_Step);

            static const vector_t m_CodesPerVector;
            static const vector_t m_CodesPerWord;
            static const vector_t m_Mask4ModStep;
            static const vector_t m_BwVec;
            static const vector_t m_MaskDecompr;

        public:
            rra_vbp_padding_l_general(const base_t * p_Data) : m_Data(p_Data) {
                //
            }

            MSV_CXX_ATTRIBUTE_FORCE_INLINE
            vector_t get(const vector_t & p_Positions) {
                using namespace vectorlib;

                // @todo Replace the division by a multiplication with the
                // inverse, if possible.
                return bitwise_and<t_ve>(
                        shift_right_individual<t_ve>::apply(
                                gather<
                                        t_ve,
                                        iov::UNALIGNED,
                                        vector_base_t_granularity::value
                                >(
                                        m_Data,
                                        // @todo `bitwise_or` would be possible
                                        // here, too (maybe more efficient).
                                        add<t_ve>::apply(
                                                shift_left<t_ve>::apply(
                                                        vectorlib::div<
                                                                t_ve
                                                        >::apply(
                                                                p_Positions,
                                                                m_CodesPerVector
                                                        ),
                                                        m_Shift4DivStep
                                                ),
                                                bitwise_and<t_ve>(
                                                        p_Positions,
                                                        m_Mask4ModStep
                                                )
                                        )
                                ),
                                mul<t_ve>::apply(
                                        m_BwVec,
                                        mod<t_ve>::apply(
                                                shift_right<t_ve>::apply(
                                                        p_Positions,
                                                        m_Shift4DivStep
                                                ),
                                                m_CodesPerWord
                                        )
                                )
                        ),
                        m_MaskDecompr
                );
            }
        };

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_padding_l_general<
                t_ve, t_Bw, t_Step
        >::m_CodesPerVector = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(vbp_padding_l<t_Bw, t_Step>::m_BlockSize);

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_padding_l_general<
                t_ve, t_Bw, t_Step
        >::m_CodesPerWord = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(vbp_padding_l<t_Bw, t_Step>::m_BlockSize / t_Step);

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_padding_l_general<
                t_ve, t_Bw, t_Step
        >::m_Mask4ModStep = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(mask_for_mod(t_Step));

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_padding_l_general<
                t_ve, t_Bw, t_Step
        >::m_BwVec = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(t_Bw);

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_padding_l_general<
                t_ve, t_Bw, t_Step
        >::m_MaskDecompr = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(bitwidth_max<typename t_ve::base_t>(t_Bw));

        /**
         * @brief An implementation of the `random_read_access` interface for
         * the layout `vbp_padding_l` which works only if the number of code
         * words per 64-bit word is a power of two for the given bit width,
         * but is more efficient than `rra_vbp_padding_l_general`.
         */
        template<class t_vector_extension, unsigned t_Bw, unsigned t_Step>
        class rra_vbp_padding_l_special {
            using t_ve = t_vector_extension;
            IMPORT_VECTOR_BOILER_PLATE(t_ve)

            const base_t * const m_Data;

            static const unsigned m_Shift4DivStep = shift_for_muldiv(t_Step);
            static const unsigned m_Shift4DivCodesPerVector =
                    shift_for_muldiv(vbp_padding_l<t_Bw, t_Step>::m_BlockSize);
            
            static const vector_t m_Mask4ModCodesPerWord;
            static const vector_t m_Mask4ModStep;
            static const vector_t m_BwVec;
            static const vector_t m_MaskDecompr;

        public:
            rra_vbp_padding_l_special(const base_t * p_Data) : m_Data(p_Data) {
                //
            }

            MSV_CXX_ATTRIBUTE_FORCE_INLINE
            vector_t get(const vector_t & p_Positions) {
                using namespace vectorlib;

                return bitwise_and<t_ve>(
                        shift_right_individual<t_ve>::apply(
                                gather<
                                        t_ve,
                                        iov::UNALIGNED,
                                        vector_base_t_granularity::value
                                >(
                                        m_Data,
                                        // @todo `bitwise_or` would be possible
                                        // here, too (maybe more efficient).
                                        add<t_ve>::apply(
                                                shift_left<t_ve>::apply(
                                                        shift_right<t_ve>::apply(
                                                                p_Positions,
                                                                m_Shift4DivCodesPerVector
                                                        ),
                                                        m_Shift4DivStep
                                                ),
                                                bitwise_and<t_ve>(
                                                        p_Positions,
                                                        m_Mask4ModStep
                                                )
                                        )
                                ),
                                mul<t_ve>::apply(
                                        m_BwVec,
                                        bitwise_and<t_ve>(
                                                shift_right<t_ve>::apply(
                                                        p_Positions,
                                                        m_Shift4DivStep
                                                ),
                                                m_Mask4ModCodesPerWord
                                        )
                                )
                        ),
                        m_MaskDecompr
                );
            }
        };

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_padding_l_special<
                t_ve, t_Bw, t_Step
        >::m_Mask4ModCodesPerWord = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(mask_for_mod(vbp_padding_l<t_Bw, t_Step>::m_BlockSize / t_Step));

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_padding_l_special<
                t_ve, t_Bw, t_Step
        >::m_Mask4ModStep = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(mask_for_mod(t_Step));

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_padding_l_special<
                t_ve, t_Bw, t_Step
        >::m_BwVec = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(t_Bw);

        template<class t_ve, unsigned t_Bw, unsigned t_Step>
        const typename t_ve::vector_t rra_vbp_padding_l_special<
                t_ve, t_Bw, t_Step
        >::m_MaskDecompr = vectorlib::set1<
                t_ve, t_ve::vector_helper_t::granularity::value
        >(bitwidth_max<typename t_ve::base_t>(t_Bw));
    
    }
    
    template<class t_vector_extension, unsigned t_Bw, unsigned t_Step>
    class random_read_access<t_vector_extension, vbp_padding_l<t_Bw, t_Step> > {
    public:
        // Alias to the most efficient implementation depending on t_Bw.
        using type = typename std::conditional<
                // If the number of code words per 64-bit word is a power of
                // two, we can use an efficient implementation.
                // Bit widths: 1, 2, 4, 8, 13-16, 22-64
                is_power_of_two(vbp_padding_l<t_Bw, t_Step>::m_BlockSize),
                _random_read_access_variants::rra_vbp_padding_l_special<
                        t_vector_extension, t_Bw, t_Step
                >,
                // Otherwise, we have to use the less efficient general
                // implementation.
                // Bit widths: 3, 5-7, 9-12, 17-21
                _random_read_access_variants::rra_vbp_padding_l_general<
                        t_vector_extension, t_Bw, t_Step
                >
        >::type;
    };
    
}

#endif //MORPHSTORE_CORE_MORPHING_VBP_PADDING_H