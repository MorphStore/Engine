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
 * @file otf_morphing_example_1.cpp
 * @brief An example micro benchmark of different degrees of integrating 
 * compression into the select-operator.
 * 
 * This micro benchmark compares two state-of-the-art operators (purely
 * uncompressed processing and a specialized operator) to two of our novel
 * enhanced operators (on-the-fly de/re-compression and on-the-fly morphing).
 * 
 * The experiment can be run using either AVX2 or AVX-512.
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/vbp.h>
#include <core/operators/general_vectorized/select_compr.h>
#include <core/operators/scalar/select_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <core/utils/variant_executor.h>
#include <vector/type_helper.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <functional>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vectorlib;

using ps_scalar = scalar<v64<uint64_t>>;
#ifdef AVXTWO
using ps_avx2 = avx2<v256<uint64_t>>;
#endif
#ifdef AVX512
using ps_avx512 = avx512<v512<uint64_t>>;
#endif

namespace morphstore {
    
    // The implementations of the specialized operator and the on-the-fly
    // morphing operator are experimental. For instance, they cannot process an
    // input column containing an arbitrary number of data elements. Therefore,
    // these implementations do not reside in one of the operators-headers, but
    // are written here.
    
    // ************************************************************************
    // Template declaration for an alternative select-operator
    // ************************************************************************
    // This is the same template declaration as for the "normal"
    // select-operator (my_select_wit_t). However, since our current template
    // parameters do not allow to distinguish between different specializations
    // such as on-the-fly de/re-compression vs. on-the-fly morphing, we use a
    // fresh struct here.
    
    template<
            template<class, int> class t_compare,
            class t_vector_extension,
            class t_out_pos_f,
            class t_in_data_f
    >
    struct select_alternative_t {
        const column<t_out_pos_f> * apply(
                const column<t_in_data_f> * inDataCol,
                uint64_t val,
                size_t outPosCountEstimate = 0
        ) = delete;
    };
    
    // ************************************************************************
    // Operator core for BitWeaving/H (==, 4 bit)
    // ************************************************************************
    // Used by the specialized operator and the on-the-fly morphing operator.
    
    // ------------------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------------------
    
    /**
     * @brief Extracts the most significant bit (MSB) of each byte in the given
     * vector.
     */
    template<class t_vector_extension>
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    uint64_t get_msbs_of_bytes(
            const typename t_vector_extension::vector_t & vec
    );

#ifdef AVXTWO
    template<>
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    uint64_t get_msbs_of_bytes<ps_avx2>(
            const typename ps_avx2::vector_t & vec
    ) {
        return static_cast<uint32_t>(_mm256_movemask_epi8(vec));
    }
#endif

#ifdef AVX512
    template<>
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    uint64_t get_msbs_of_bytes<ps_avx512>(
            const typename ps_avx512::vector_t & vec
    ) {
        return _mm512_cmpgt_epu8_mask(vec, _mm512_set1_epi8(0x7f));
    }
#endif
    
    template<class t_vector_extension>
    const uint64_t pextMask;
    
#ifdef AVXTWO
    template<>
    const uint64_t pextMask<ps_avx2> = 0x01010101;
#endif
#ifdef AVX512
    template<>
    const uint64_t pextMask<ps_avx512> = 0x0101010101010101ull;
#endif
    
    // ------------------------------------------------------------------------
    // Operator core for BitWeaving/H (==, 4 Bit)
    // ------------------------------------------------------------------------
    
    #define OUTPUT_POSITIONS(maskExpr) \
        tmpMask = maskExpr; \
        compressstore<t_ve, iov::UNALIGNED, vector_size_bit::value>( \
            outPos, posVec, tmpMask \
        ); \
        outPos += count_matches<t_ve>::apply(tmpMask); \
        posVec = add<t_ve, vector_base_t_granularity::value>::apply( \
            posVec, posIncVec \
        );
    
    template<class t_vector_extension>
    struct core_bwh_eq_4bit_t {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static void apply(
                const vector_t & inVec, // the input data
                const vector_t & predVec, // the constant of the comparison
                const vector_t & maskPayloadVec, // a bitmask where all non-delimiter bits are 1
                const vector_t & maskDelimVec, // a bitmask where all delimiter bits are 1
                vector_t & posVec, // the output positions
                const vector_t & posIncVec, // vector by which to increment the output positions
                base_t * & outPos // output buffer
        ) {
            // BitWeaving/H-style ==-comparison.
            const vector_t cmpVec = bitwise_andnot<t_ve, vector_size_bit::value>(
                    add<t_ve, vector_base_t_granularity::value>::apply(
                            bitwise_xor<t_ve, vector_size_bit::value>(
                                    inVec, predVec
                            ),
                            maskPayloadVec
                    ),
                    maskDelimVec
            );
            
            // Extract the delimiter bits (the bits whose positions are 
            // multiples of four).
            const uint64_t maskOdd = get_msbs_of_bytes<t_ve>(cmpVec);
            const uint64_t maskEven = get_msbs_of_bytes<t_ve>(
                    shift_left<t_ve, vector_base_t_granularity::value>::apply(
                            cmpVec, 4
                    )
            );
            
            // Write out positions.
            if(maskOdd | maskEven) {
                // There are matches in the given data vector inVec.
                vector_mask_t tmpMask;
                // @todo Manually unroll this?
                for(unsigned i = 0; i < 8; i++) {
                    OUTPUT_POSITIONS(_pext_u64(maskEven, pextMask<t_ve> << i))
                    OUTPUT_POSITIONS(_pext_u64(maskOdd , pextMask<t_ve> << i))
                }
            }
            else {
                // There are no matches in the given data vector inVec.
                posVec = add<t_ve, vector_base_t_granularity::value>::apply(
                        posVec,
                        shift_left<t_ve, vector_base_t_granularity::value>::apply(
                                posIncVec, 4
                        )
                );
            }
        }
    };
    
    // ************************************************************************
    // Specialized select-operator for BW/H (==, 4 bit) on 4-bit packed data
    // ************************************************************************
    // Requires no morphing.
    
    template<class t_vector_extension>
    struct select_alternative_t<
            equal,
            t_vector_extension,
            uncompr_f,
            static_vbp_f<vbp_l<
                    4,
                    t_vector_extension::vector_helper_t::element_count::value>
            >
    > {
        using t_out_pos_f = uncompr_f;
        using t_in_data_f = static_vbp_f<vbp_l<
                4,
                t_vector_extension::vector_helper_t::element_count::value>
        >;
        using ve64 = t_vector_extension;
        using ve8 = typename TypeHelper<t_vector_extension, uint8_t>::newbasetype;
        IMPORT_VECTOR_BOILER_PLATE(ve64)
        IMPORT_VECTOR_BOILER_PLATE_PREFIX(ve8, b8_)
        
        static const column<t_out_pos_f> * apply(
                const column<t_in_data_f> * inDataCol,
                uint64_t pred,
                MSV_CXX_ATTRIBUTE_PPUNUSED size_t est = 0
        ) {
            const base_t * inData = inDataCol->get_data();
            const size_t inDataCount = inDataCol->get_count_values();
            const base_t * const inDataEnd = reinterpret_cast<const base_t *>(
                    reinterpret_cast<const uint8_t *>(inData) +
                    inDataCol->get_size_compr_byte()
            );
            
            auto outPosCol = new column<t_out_pos_f>(
                    inDataCount * sizeof(uint64_t) // pessimistic estimate
            );
            base_t * outPos = outPosCol->get_data();
            const base_t * const initOutPos = outPos;
            
            // Some helper vector registers.
            const vector_t predVec =
                    set1<ve8, b8_vector_base_t_granularity::value>(
                            (pred << 4) | pred
                    );
            const vector_t maskPayloadVec =
                    set1<ve8, b8_vector_base_t_granularity::value>(
                            0b01110111
                    );
            const vector_t maskDelimVec =
                    set1<ve8, b8_vector_base_t_granularity::value>(
                            0b10001000
                    );
            vector_t posVec =
                    set_sequence<ve64, vector_base_t_granularity::value>(
                            0, 1
                    );
            const vector_t posIncVec =
                    set1<ve64, vector_base_t_granularity::value>(
                            vector_element_count::value
                    );
            
            // Repeatedly call the operator core for all input vectors.
            for(; inData < inDataEnd; inData += vector_element_count::value)
                core_bwh_eq_4bit_t<ve64>::apply(
                        // input data
                        load<ve64, iov::ALIGNED, vector_size_bit::value>(inData),
                        // helpers
                        predVec, maskPayloadVec, maskDelimVec,
                        posVec, posIncVec,
                        // output buffer
                        outPos
                );
            
            const size_t outPosCount = outPos - initOutPos;
            outPosCol->set_meta_data(
                    outPosCount, outPosCount * sizeof(uint64_t)
            );
            
            return outPosCol;
        }
    };
    
    
    // ************************************************************************
    // On-the-fly morphing operator for BW/H (==, 4 bit) on 3-bit packed data
    // ************************************************************************
    // Requires morphing from 3-bit packed to 4-bit packed.
    
    // ------------------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------------------

    template<class t_vector_extension>
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    typename t_vector_extension::vector_t permute(
            const typename t_vector_extension::vector_t & vec,
            const typename t_vector_extension::vector_t & mask
    );
    
#ifdef AVXTWO
    template<>
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    typename ps_avx2::vector_t permute<ps_avx2>(
            const typename ps_avx2::vector_t & vec,
            const typename ps_avx2::vector_t & mask
    ) {
        return _mm256_shuffle_epi8(vec, mask);
    }
#endif
    
#ifdef AVX512
    template<>
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    typename ps_avx512::vector_t permute<ps_avx512>(
            const typename ps_avx512::vector_t & vec,
            const typename ps_avx512::vector_t & mask
    ) {
        return _mm512_shuffle_epi8(vec, mask);
    }
#endif
    
    // ------------------------------------------------------------------------
    // Direct integer morphing algorithm (from 3 bit to 4 bit packed data)
    // ------------------------------------------------------------------------
    
    template<class t_vector_extension>
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    void morph3to4_one_cycle(
            // For the morphing algorithm
            const typename t_vector_extension::vector_t & origVec,
            const typename t_vector_extension::vector_t & permMaskVec,
            const typename t_vector_extension::vector_t & shiftVec,
            const typename t_vector_extension::vector_t & mask3Vec0,
            const typename t_vector_extension::vector_t & mask3Vec1,
            const typename t_vector_extension::vector_t & mask3Vec2,
            const typename t_vector_extension::vector_t & mask3Vec3,
            // For the BW/H core
            const typename t_vector_extension::vector_t & predVec,
            const typename t_vector_extension::vector_t & maskPayloadVec,
            const typename t_vector_extension::vector_t & maskDelimVec,
            typename t_vector_extension::vector_t & posVec,
            const typename t_vector_extension::vector_t & posIncVec,
            typename t_vector_extension::base_t * & outPos
    ) {
        using ve64 = t_vector_extension;
        using ve16 = typename TypeHelper<t_vector_extension, uint16_t>::newbasetype;
        IMPORT_VECTOR_BOILER_PLATE_PREFIX(ve64, b64_)
        IMPORT_VECTOR_BOILER_PLATE_PREFIX(ve16, b16_)
        
        const b64_vector_t vec = shift_right_individual<
                ve16, b16_vector_base_t_granularity::value
        >::apply(permute<ve64>(origVec, permMaskVec), shiftVec);
        
        b64_vector_t outVec = bitwise_and<ve64, b64_vector_size_bit::value>(
                vec, mask3Vec0
        );
        // Actually, it does not matter, in which elements we shift, could also
        // be 64 bit.
        outVec = bitwise_or<ve64, b64_vector_size_bit::value>(
                outVec,
                bitwise_and<ve64, b64_vector_size_bit::value>(
                        shift_left<
                                ve16, b16_vector_base_t_granularity::value
                        >::apply(vec, 1),
                        mask3Vec1
                )
        );
        outVec = bitwise_or<ve64, b64_vector_size_bit::value>(
                outVec,
                bitwise_and<ve64, b64_vector_size_bit::value>(
                        shift_left<
                                ve16, b16_vector_base_t_granularity::value
                        >::apply(vec, 2),
                        mask3Vec2
                )
        );
        outVec = bitwise_or<ve64, b64_vector_size_bit::value>(
                outVec,
                bitwise_and<ve64, b64_vector_size_bit::value>(
                        shift_left<
                                ve16, b16_vector_base_t_granularity::value
                        >::apply(vec, 3),
                        mask3Vec3
                )
        );
        
        core_bwh_eq_4bit_t<ve64>::apply(
                outVec,
                        
                predVec, maskPayloadVec, maskDelimVec,
                posVec, posIncVec,

                outPos
        );
    }
    
    // ------------------------------------------------------------------------
    // On-the-fly morphing operator
    // ------------------------------------------------------------------------
    
    template<class t_ve>
    const typename t_ve::vector_t permMaskVec;
    
#ifdef AVXTWO
    template<>
    const typename ps_avx2::vector_t permMaskVec<ps_avx2> =
            set<ps_avx2, ps_avx2::vector_helper_t::granularity::value>(
                    0x0d0c0c0b0a090908ull,
                    0x0504040302010100ull,
                    0x0d0c0c0b0a090908ull,
                    0x0504040302010100ull
            );
#endif
    
#ifdef AVX512
    template<>
    const typename ps_avx512::vector_t permMaskVec<ps_avx512> =
            set<ps_avx512, ps_avx512::vector_helper_t::granularity::value>(
                    0x0d0c0c0b0a090908ull,
                    0x0504040302010100ull,
                    0x0d0c0c0b0a090908ull,
                    0x0504040302010100ull,

                    0x0d0c0c0b0a090908ull,
                    0x0504040302010100ull,
                    0x0d0c0c0b0a090908ull,
                    0x0504040302010100ull
            );
#endif
    
    template<class t_vector_extension>
    struct select_alternative_t<
            equal,
            t_vector_extension,
            uncompr_f,
            static_vbp_f<vbp_l<
                    3,
                    t_vector_extension::vector_helper_t::element_count::value
            > >
    > {
        using t_out_pos_f = uncompr_f;
        using t_in_data_f = static_vbp_f<vbp_l<
                3,
                t_vector_extension::vector_helper_t::element_count::value
        > >;
        using ve64 = t_vector_extension;
        using ve16 = typename TypeHelper<t_vector_extension, uint16_t>::newbasetype;
        using ve8 = typename TypeHelper<t_vector_extension, uint8_t>::newbasetype;
        IMPORT_VECTOR_BOILER_PLATE(ve64)
        IMPORT_VECTOR_BOILER_PLATE_PREFIX(ve8, b8_)
        IMPORT_VECTOR_BOILER_PLATE_PREFIX(ve16, b16_)
        
        static const column<t_out_pos_f> * apply(
                const column<t_in_data_f> * inDataCol,
                uint64_t pred,
                MSV_CXX_ATTRIBUTE_PPUNUSED size_t est = 0
        ) {
            const base_t * inData = inDataCol->get_data();
            const size_t inDataCount = inDataCol->get_count_values();
            const base_t * const inDataEnd = reinterpret_cast<const base_t *>(
                    reinterpret_cast<const uint8_t *>(inData) +
                    inDataCol->get_size_compr_byte()
            );
            
            auto outPosCol = new column<t_out_pos_f>(
                    inDataCount * sizeof(uint64_t) // pessimistic estimate
            );
            base_t * outPos = outPosCol->get_data();
            const base_t * const initOutPos = outPos;
            
            const size_t vecElCount = vector_element_count::value;
            
            const vector_t predVec =
                    set1<ve8, b8_vector_base_t_granularity::value>(
                            (pred << 4) | pred
                    );
            const vector_t maskPayloadVec =
                    set1<ve8, b8_vector_base_t_granularity::value>(0b01110111);
            const vector_t maskDelimVec =
                    set1<ve8, b8_vector_base_t_granularity::value>(0b10001000);
            vector_t posVec =
                    set_sequence<ve64, vector_base_t_granularity::value>(0, 1);
            const vector_t posIncVec =
                    set1<ve64, vector_base_t_granularity::value>(vecElCount);
            
            const vector_t shiftVec =
                    set1<ve64, vector_base_t_granularity::value>(
                            0x0004000000040000ull
                    );
            const vector_t mask3Vec0 =
                    set1<ve16, b16_vector_base_t_granularity::value>(
                            0b0000000000000111
                    );
            const vector_t mask3Vec1 =
                    set1<ve16, b16_vector_base_t_granularity::value>(
                            0b0000000001110000
                    );
            const vector_t mask3Vec2 =
                    set1<ve16, b16_vector_base_t_granularity::value>(
                            0b0000011100000000
                    );
            const vector_t mask3Vec3 =
                    set1<ve16, b16_vector_base_t_granularity::value>(
                            0b0111000000000000
                    );
            
            for(; inData < inDataEnd; inData += 3 * vecElCount) {
                const vector_t inVec1 = load<ve64, iov::ALIGNED, vector_size_bit::value>(inData);
                const vector_t inVec2 = load<ve64, iov::ALIGNED, vector_size_bit::value>(inData + vecElCount);
                const vector_t inVec3 = load<ve64, iov::ALIGNED, vector_size_bit::value>(inData + 2 * vecElCount);
                
                const vector_t v1 = inVec1;
                const vector_t v2 = bitwise_or<ve64, vector_size_bit::value>(
                        shift_right<ve64, vector_base_t_granularity::value>::apply(inVec1, 48),
                        shift_left<ve64, vector_base_t_granularity::value>::apply(inVec2, 16)
                );
                const vector_t v3 = bitwise_or<ve64, vector_size_bit::value>(
                        shift_right<ve64, vector_base_t_granularity::value>::apply(inVec2, 32),
                        shift_left<ve64, vector_base_t_granularity::value>::apply(inVec3, 32)
                );
                const vector_t v4 = shift_right<ve64, vector_base_t_granularity::value>::apply(inVec3, 16);
                
                morph3to4_one_cycle<ve64>(
                        // For the morphing algorithm
                        v1,
                        permMaskVec<ve64>,
                        shiftVec, mask3Vec0, mask3Vec1, mask3Vec2, mask3Vec3,
                        // For the BW/H core
                        predVec, maskPayloadVec, maskDelimVec, posVec, posIncVec,
                        outPos
                );
                morph3to4_one_cycle<ve64>(
                        // For the morphing algorithm
                        v2,
                        permMaskVec<ve64>,
                        shiftVec, mask3Vec0, mask3Vec1, mask3Vec2, mask3Vec3,
                        // For the BW/H core
                        predVec, maskPayloadVec, maskDelimVec, posVec, posIncVec,
                        outPos
                );
                morph3to4_one_cycle<ve64>(
                        // For the morphing algorithm
                        v3,
                        permMaskVec<ve64>,
                        shiftVec, mask3Vec0, mask3Vec1, mask3Vec2, mask3Vec3,
                        // For the BW/H core
                        predVec, maskPayloadVec, maskDelimVec, posVec, posIncVec,
                        outPos
                );
                morph3to4_one_cycle<ve64>(
                        // For the morphing algorithm
                        v4,
                        permMaskVec<ve64>,
                        shiftVec, mask3Vec0, mask3Vec1, mask3Vec2, mask3Vec3,
                        // For the BW/H core
                        predVec, maskPayloadVec, maskDelimVec, posVec, posIncVec,
                        outPos
                );
            }
            
            const size_t outPosCount = outPos - initOutPos;
            outPosCol->set_meta_data(
                    outPosCount, outPosCount * sizeof(uint64_t)
            );
            
            return outPosCol;
        }
    };
    
} // namespace morphstore


#define MAKE_VARIANT(vector_extension, op, comp, out_pos_f, in_data_f, opClass) { \
    new typename varex_t::operator_wrapper::template for_output_formats<out_pos_f>::template for_input_formats<in_data_f>( \
        &op<comp, vector_extension, out_pos_f, in_data_f>::apply \
    ), \
    STR_EVAL_MACROS(vector_extension), \
    STR_EVAL_MACROS(out_pos_f), \
    STR_EVAL_MACROS(in_data_f), \
    STR_EVAL_MACROS(op), \
    opClass \
}


// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    // ========================================================================
    // Creation of the variant executor.
    // ========================================================================
    
    using varex_t = variant_executor_helper<1, 1, uint64_t, size_t>::type
        ::for_variant_params<
                std::string, std::string, std::string, std::string, std::string
        >
        ::for_setting_params<size_t, double>;
    varex_t varex(
            {"predicate", "estimate"},
            {
                "vector_extension",
                "out_pos_f", "in_data_f",
                "operator_name", "operator_class"
            },
            {"countValues", "sel"}
    );

    // ========================================================================
    // Variant execution for each setting.
    // ========================================================================
    
    // 4 GiB uncompressed, 256 MiB @4-bit, 192 MiB @3-bit.
    const size_t countValues = 512 * 1024 * 1024ull;
    std::vector<double> sels = {
        static_cast<double>(1) / 10000,
    };
    for(double sel : sels) {
        // --------------------------------------------------------------------
        // Data generation.
        // --------------------------------------------------------------------

        varex.print_datagen_started();

        const column<uncompr_f> * inDataCol = generate_exact_number(
                countValues, static_cast<size_t>(sel * countValues), 0, 7, false
        );
        varex.print_datagen_done();

        // --------------------------------------------------------------------
        // Variant generation.
        // --------------------------------------------------------------------

        std::vector<varex_t::variant_t> variants = {
            // Reference and cold start.
            MAKE_VARIANT(ps_scalar, select_t, std::equal_to, uncompr_f, uncompr_f, "uncompressed"),
#ifdef AVX512
            MAKE_VARIANT(ps_avx512, my_select_wit_t     , equal, uncompr_f, uncompr_f, "uncompressed"),
            MAKE_VARIANT(ps_avx512, my_select_wit_t     , equal, uncompr_f, SINGLE_ARG(static_vbp_f<vbp_l<3, 8>>), "otf de/re-compression"),
            MAKE_VARIANT(ps_avx512, select_alternative_t, equal, uncompr_f, SINGLE_ARG(static_vbp_f<vbp_l<4, 8>>), "specialized"),
            MAKE_VARIANT(ps_avx512, select_alternative_t, equal, uncompr_f, SINGLE_ARG(static_vbp_f<vbp_l<3, 8>>), "otf morphing"),
#elif defined(AVXTWO)
            MAKE_VARIANT(ps_avx2  , my_select_wit_t     , equal, uncompr_f, uncompr_f, "uncompressed"),
            MAKE_VARIANT(ps_avx2  , my_select_wit_t     , equal, uncompr_f, SINGLE_ARG(static_vbp_f<vbp_l<3, 4>>), "otf de/re-compression"),
            MAKE_VARIANT(ps_avx2  , select_alternative_t, equal, uncompr_f, SINGLE_ARG(static_vbp_f<vbp_l<4, 4>>), "specialized"),
            MAKE_VARIANT(ps_avx2  , select_alternative_t, equal, uncompr_f, SINGLE_ARG(static_vbp_f<vbp_l<3, 4>>), "otf morphing"),
#endif
        };

        // --------------------------------------------------------------------
        // Variant execution.
        // --------------------------------------------------------------------

        varex.execute_variants(variants, countValues, sel, inDataCol, 0, 0);

        delete inDataCol;
    }
    
    varex.done();
    
    return !varex.good();
}