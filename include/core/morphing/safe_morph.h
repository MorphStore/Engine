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
 * @file safe_morph.h
 * @brief A utility for using the morph-operator without specifying a vector
 * extension.
 * 
 * This utility is meant to be used in MorphStore infrastructure code such as
 * `variant_executor` and `column_cache`, where it is sometimes necessary to
 * morph a column from some source format to some destination format using
 * *any* vector extension as long as it works.
 * 
 * `safe_morph` should not be used in query or operator code, because it does
 * not expose the vector extension to be used to the caller and may, thus,
 * result in unexpected performance behavior. Use the normal `morph` instead.
 * 
 * The safe morph-operator `safe_morph_t` should be (partially) template
 * specialized for any (required) combination of source and destination
 * formats. These specializations should delegate to the normal morph-operator,
 * while hardcoding the vector extension to be used.
 */

#ifndef MORPHSTORE_CORE_MORPHING_SAFE_MORPH_H
#define MORPHSTORE_CORE_MORPHING_SAFE_MORPH_H

#include <core/morphing/delta.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/for.h>
#include <core/morphing/k_wise_ns.h>
#include <core/morphing/morph.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/type_packing.h>
#include <core/morphing/vbp.h>
#include <core/morphing/vbp_padding.h>
#include <core/storage/column.h>
#include <vector/vector_extension_structs.h>

#include <cstdint>

namespace morphstore {

    // ************************************************************************
    // Interface
    // ************************************************************************

    // Struct.
    template<class t_dst_f, class t_src_f>
    struct safe_morph_t {
        static
        const column<t_dst_f> *
        apply(const column<t_src_f> * inCol) = delete;
    };

    // Specialization for equal source and destination format.
    template<class t_f>
    struct safe_morph_t<t_f, t_f> {
        static
        const column<t_f> *
        apply(const column<t_f> * inCol) {
            return morph<vectorlib::scalar<vectorlib::v64<uint64_t>>, t_f, t_f>(
                    inCol
            );
        };
    };

    // Convenience function.
    template<class t_dst_f, class t_src_f>
    const column<t_dst_f> * safe_morph(const column<t_src_f> * inCol) {
        return safe_morph_t<t_dst_f, t_src_f>::apply(inCol);
    }

    // ************************************************************************
    // Specializations
    // ************************************************************************

    // ------------------------------------------------------------------------
    // Format static_vbp_f
    // ------------------------------------------------------------------------
    
    // The step template parameter of the format static_vbp_f corresponds to
    // the vector extension that naturally fits for working with this format.
    
    // Compression

    #define MAKE_SAFE_MORPH_STATIC_VBP_COMPR(vector_extension, layout) \
        template<unsigned t_bw> \
        struct safe_morph_t< \
                static_vbp_f<layout< \
                        t_bw, \
                        vector_extension::vector_helper_t::element_count::value \
                > >, \
                uncompr_f \
        > { \
            using dst_f = static_vbp_f<layout< \
                    t_bw, \
                    vector_extension::vector_helper_t::element_count::value \
            > >; \
            using src_f = uncompr_f; \
             \
            static \
            const column<dst_f> * \
            apply(const column<src_f> * inCol) { \
                return morph<vector_extension, dst_f, src_f>(inCol); \
            } \
        };

    MAKE_SAFE_MORPH_STATIC_VBP_COMPR(vectorlib::scalar<vectorlib::v64<uint64_t>>, vbp_l)
    MAKE_SAFE_MORPH_STATIC_VBP_COMPR(vectorlib::scalar<vectorlib::v64<uint64_t>>, vbp_padding_l)
#ifdef SSE
    MAKE_SAFE_MORPH_STATIC_VBP_COMPR(vectorlib::sse<vectorlib::v128<uint64_t>>, vbp_l)
    MAKE_SAFE_MORPH_STATIC_VBP_COMPR(vectorlib::sse<vectorlib::v128<uint64_t>>, vbp_padding_l)
#endif
#ifdef AVXTWO
    MAKE_SAFE_MORPH_STATIC_VBP_COMPR(vectorlib::avx2<vectorlib::v256<uint64_t>>, vbp_l)
    MAKE_SAFE_MORPH_STATIC_VBP_COMPR(vectorlib::avx2<vectorlib::v256<uint64_t>>, vbp_padding_l)
#endif
#ifdef AVX512
    MAKE_SAFE_MORPH_STATIC_VBP_COMPR(vectorlib::avx512<vectorlib::v512<uint64_t>>, vbp_l)
    MAKE_SAFE_MORPH_STATIC_VBP_COMPR(vectorlib::avx512<vectorlib::v512<uint64_t>>, vbp_padding_l)
#endif

    #undef MAKE_SAFE_MORPH_STATIC_VBP_COMPR

    // Decompression.

    #define MAKE_SAFE_MORPH_STATIC_VBP_DECOMPR(vector_extension, layout) \
    template<unsigned t_bw> \
    struct safe_morph_t< \
            uncompr_f, \
            static_vbp_f<layout< \
                    t_bw, \
                    vector_extension::vector_helper_t::element_count::value \
            > > \
    > { \
        using dst_f = uncompr_f; \
        using src_f = static_vbp_f<layout< \
                t_bw, \
                vector_extension::vector_helper_t::element_count::value \
        > >; \
         \
        static \
        const column<dst_f> * \
        apply(const column<src_f> * inCol) { \
            return morph<vector_extension, dst_f, src_f>(inCol); \
        } \
    };

    MAKE_SAFE_MORPH_STATIC_VBP_DECOMPR(vectorlib::scalar<vectorlib::v64<uint64_t>>, vbp_l)
    MAKE_SAFE_MORPH_STATIC_VBP_DECOMPR(vectorlib::scalar<vectorlib::v64<uint64_t>>, vbp_padding_l)
#ifdef SSE
    MAKE_SAFE_MORPH_STATIC_VBP_DECOMPR(vectorlib::sse<vectorlib::v128<uint64_t>>, vbp_l)
    MAKE_SAFE_MORPH_STATIC_VBP_DECOMPR(vectorlib::sse<vectorlib::v128<uint64_t>>, vbp_padding_l)
#endif
#ifdef AVXTWO
    MAKE_SAFE_MORPH_STATIC_VBP_DECOMPR(vectorlib::avx2<vectorlib::v256<uint64_t>>, vbp_l)
    MAKE_SAFE_MORPH_STATIC_VBP_DECOMPR(vectorlib::avx2<vectorlib::v256<uint64_t>>, vbp_padding_l)
#endif
#ifdef AVX512
    MAKE_SAFE_MORPH_STATIC_VBP_DECOMPR(vectorlib::avx512<vectorlib::v512<uint64_t>>, vbp_l)
    MAKE_SAFE_MORPH_STATIC_VBP_DECOMPR(vectorlib::avx512<vectorlib::v512<uint64_t>>, vbp_padding_l)
#endif

    #undef MAKE_SAFE_MORPH_STATIC_VBP_DECOMPR

    // ------------------------------------------------------------------------
    // Format dynamic_vbp_f
    // ------------------------------------------------------------------------
    
    // The step template parameter of the format dynamic_vbp_f corresponds to
    // the vector extension that naturally fits for working with this format.
    
    // Compression
            
    // @todo It would be nice to have the template parameters t_BlockSizeLog and
    // t_PageSizeBlocks here as well, but the compiler complains then.
    #define MAKE_SAFE_MORPH_DYNAMIC_VBP_COMPR(vector_extension) \
        template<> \
        struct safe_morph_t< \
                dynamic_vbp_f< \
                        vector_extension::vector_helper_t::size_bit::value, \
                        vector_extension::vector_helper_t::size_byte::value, \
                        vector_extension::vector_helper_t::element_count::value \
                >, \
                uncompr_f \
        > { \
            using dst_f = dynamic_vbp_f< \
                    vector_extension::vector_helper_t::size_bit::value, \
                    vector_extension::vector_helper_t::size_byte::value, \
                    vector_extension::vector_helper_t::element_count::value \
            >; \
            using src_f = uncompr_f; \
             \
            static \
            const column<dst_f> * \
            apply(const column<src_f> * inCol) { \
                return morph<vector_extension, dst_f, src_f>(inCol); \
            } \
        };

    // @todo This always consumes a lot of time, even if dynamic_vbp_f in not
    // used in the respective compilation unit.
    MAKE_SAFE_MORPH_DYNAMIC_VBP_COMPR(vectorlib::scalar<vectorlib::v64<uint64_t>>)
#ifdef SSE
    MAKE_SAFE_MORPH_DYNAMIC_VBP_COMPR(vectorlib::sse<vectorlib::v128<uint64_t>>)
#endif
#ifdef AVXTWO
    MAKE_SAFE_MORPH_DYNAMIC_VBP_COMPR(vectorlib::avx2<vectorlib::v256<uint64_t>>)
#endif
#ifdef AVX512
    MAKE_SAFE_MORPH_DYNAMIC_VBP_COMPR(vectorlib::avx512<vectorlib::v512<uint64_t>>)
#endif

    #undef MAKE_SAFE_MORPH_DYNAMIC_VBP_COMPR

    // Decompression.
            
    // @todo It would be nice to have the template parameters t_BlockSizeLog and
    // t_PageSizeBlocks here as well, but the compiler complains then.
    #define MAKE_SAFE_MORPH_DYNAMIC_VBP_DECOMPR(vector_extension) \
    template<> \
    struct safe_morph_t< \
            uncompr_f, \
            dynamic_vbp_f< \
                    vector_extension::vector_helper_t::size_bit::value, \
                    vector_extension::vector_helper_t::size_byte::value, \
                    vector_extension::vector_helper_t::element_count::value \
            > \
    > { \
        using dst_f = uncompr_f; \
        using src_f = dynamic_vbp_f< \
                vector_extension::vector_helper_t::size_bit::value, \
                vector_extension::vector_helper_t::size_byte::value, \
                vector_extension::vector_helper_t::element_count::value \
        >; \
         \
        static \
        const column<dst_f> * \
        apply(const column<src_f> * inCol) { \
            return morph<vector_extension, dst_f, src_f>(inCol); \
        } \
    };

    // @todo This always consumes a lot of time, even if dynamic_vbp_f in not
    // used in the respective compilation unit.
    MAKE_SAFE_MORPH_DYNAMIC_VBP_DECOMPR(vectorlib::scalar<vectorlib::v64<uint64_t>>)
#ifdef SSE
    MAKE_SAFE_MORPH_DYNAMIC_VBP_DECOMPR(vectorlib::sse<vectorlib::v128<uint64_t>>)
#endif
#ifdef AVXTWO
    MAKE_SAFE_MORPH_DYNAMIC_VBP_DECOMPR(vectorlib::avx2<vectorlib::v256<uint64_t>>)
#endif
#ifdef AVX512
    MAKE_SAFE_MORPH_DYNAMIC_VBP_DECOMPR(vectorlib::avx512<vectorlib::v512<uint64_t>>)
#endif

    #undef MAKE_SAFE_MORPH_DYNAMIC_VBP_DECOMPR

    // ------------------------------------------------------------------------
    // Format k_wise_ns_f
    // ------------------------------------------------------------------------
            
#ifdef SSE
    template<>
    struct safe_morph_t<
            k_wise_ns_f<
                    vectorlib::sse<
                            vectorlib::v128<uint64_t>
                    >::vector_helper_t::element_count::value
            >,
            uncompr_f
    > {
        using dst_f = k_wise_ns_f<
                vectorlib::sse<
                        vectorlib::v128<uint64_t>
                >::vector_helper_t::element_count::value
        >;
        using src_f = uncompr_f;
        
        static
        const column<dst_f> *
        apply(const column<src_f> * inCol) {
            return morph<vectorlib::sse<vectorlib::v128<uint64_t>>, dst_f, src_f>(inCol);
        };
    };
            
    template<>
    struct safe_morph_t<
            uncompr_f,
            k_wise_ns_f<
                    vectorlib::sse<
                            vectorlib::v128<uint64_t>
                    >::vector_helper_t::element_count::value
            >
    > {
        using dst_f = uncompr_f;
        using src_f = k_wise_ns_f<
                vectorlib::sse<
                        vectorlib::v128<uint64_t>
                >::vector_helper_t::element_count::value
        >;
        
        static
        const column<dst_f> *
        apply(const column<src_f> * inCol) {
            return morph<vectorlib::sse<vectorlib::v128<uint64_t>>, dst_f, src_f>(inCol);
        };
    };
#endif
    
    // ------------------------------------------------------------------------
    // Format delta_f
    // ------------------------------------------------------------------------
    
    // The step template parameter of the format delta_f corresponds to
    // the vector extension that naturally fits for working with this format.
    
    // Compression
            
    #define MAKE_SAFE_MORPH_DELTA_COMPR(vector_extension) \
    template<size_t t_BlockSizeLog, class t_inner_f> \
    struct safe_morph_t<delta_f<t_BlockSizeLog, vector_extension::vector_helper_t::element_count::value, t_inner_f>, uncompr_f> { \
        using dst_f = delta_f<t_BlockSizeLog, vector_extension::vector_helper_t::element_count::value, t_inner_f>; \
        using src_f = uncompr_f; \
         \
        static \
        const column<dst_f> * \
        apply(const column<src_f> * inCol) { \
            return morph<vector_extension, dst_f, src_f>(inCol); \
        }; \
    };
    
    MAKE_SAFE_MORPH_DELTA_COMPR(vectorlib::scalar<vectorlib::v64<uint64_t>>)
#ifdef SSE
    MAKE_SAFE_MORPH_DELTA_COMPR(vectorlib::sse<vectorlib::v128<uint64_t>>)
#endif
#ifdef AVXTWO
    MAKE_SAFE_MORPH_DELTA_COMPR(vectorlib::avx2<vectorlib::v256<uint64_t>>)
#endif
#ifdef AVX512
    MAKE_SAFE_MORPH_DELTA_COMPR(vectorlib::avx512<vectorlib::v512<uint64_t>>)
#endif
            
    #undef MAKE_SAFE_MORPH_DELTA_COMPR

    // Decompression
            
    #define MAKE_SAFE_MORPH_DELTA_DECOMPR(vector_extension) \
    template<size_t t_BlockSizeLog, class t_inner_f> \
    struct safe_morph_t<uncompr_f, delta_f<t_BlockSizeLog, vector_extension::vector_helper_t::element_count::value, t_inner_f> > { \
        using dst_f = uncompr_f; \
        using src_f = delta_f<t_BlockSizeLog, vector_extension::vector_helper_t::element_count::value, t_inner_f>; \
         \
        static \
        const column<dst_f> * \
        apply(const column<src_f> * inCol) { \
            return morph<vector_extension, dst_f, src_f>(inCol); \
        }; \
    };
    
    MAKE_SAFE_MORPH_DELTA_DECOMPR(vectorlib::scalar<vectorlib::v64<uint64_t>>)
#ifdef SSE
    MAKE_SAFE_MORPH_DELTA_DECOMPR(vectorlib::sse<vectorlib::v128<uint64_t>>)
#endif
#ifdef AVXTWO
    MAKE_SAFE_MORPH_DELTA_DECOMPR(vectorlib::avx2<vectorlib::v256<uint64_t>>)
#endif
#ifdef AVX512
    MAKE_SAFE_MORPH_DELTA_DECOMPR(vectorlib::avx512<vectorlib::v512<uint64_t>>)
#endif
            
    #undef MAKE_SAFE_MORPH_DELTA_DECOMPR
    
    // ------------------------------------------------------------------------
    // Format for_f
    // ------------------------------------------------------------------------
    
    // The page size in blocks template parameter of the format for_f
    // corresponds to the vector extension that naturally fits for working with
    // this format.
    
    // Compression
            
    #define MAKE_SAFE_MORPH_FOR_COMPR(vector_extension) \
    template<size_t t_BlockSizeLog, class t_inner_f> \
    struct safe_morph_t<for_f<t_BlockSizeLog, vector_extension::vector_helper_t::element_count::value, t_inner_f>, uncompr_f> { \
        using dst_f = for_f<t_BlockSizeLog, vector_extension::vector_helper_t::element_count::value, t_inner_f>; \
        using src_f = uncompr_f; \
         \
        static \
        const column<dst_f> * \
        apply(const column<src_f> * inCol) { \
            return morph<vector_extension, dst_f, src_f>(inCol); \
        }; \
    };
    
    MAKE_SAFE_MORPH_FOR_COMPR(vectorlib::scalar<vectorlib::v64<uint64_t>>)
#ifdef SSE
    MAKE_SAFE_MORPH_FOR_COMPR(vectorlib::sse<vectorlib::v128<uint64_t>>)
#endif
#ifdef AVXTWO
    MAKE_SAFE_MORPH_FOR_COMPR(vectorlib::avx2<vectorlib::v256<uint64_t>>)
#endif
#ifdef AVX512
    MAKE_SAFE_MORPH_FOR_COMPR(vectorlib::avx512<vectorlib::v512<uint64_t>>)
#endif
            
    #undef MAKE_SAFE_MORPH_FOR_COMPR

    // Decompression
            
    #define MAKE_SAFE_MORPH_FOR_DECOMPR(vector_extension) \
    template<size_t t_BlockSizeLog, class t_inner_f> \
    struct safe_morph_t<uncompr_f, for_f<t_BlockSizeLog, vector_extension::vector_helper_t::element_count::value, t_inner_f> > { \
        using dst_f = uncompr_f; \
        using src_f = for_f<t_BlockSizeLog, vector_extension::vector_helper_t::element_count::value, t_inner_f>; \
         \
        static \
        const column<dst_f> * \
        apply(const column<src_f> * inCol) { \
            return morph<vector_extension, dst_f, src_f>(inCol); \
        }; \
    };
    
    MAKE_SAFE_MORPH_FOR_DECOMPR(vectorlib::scalar<vectorlib::v64<uint64_t>>)
#ifdef SSE
    MAKE_SAFE_MORPH_FOR_DECOMPR(vectorlib::sse<vectorlib::v128<uint64_t>>)
#endif
#ifdef AVXTWO
    MAKE_SAFE_MORPH_FOR_DECOMPR(vectorlib::avx2<vectorlib::v256<uint64_t>>)
#endif
#ifdef AVX512
    MAKE_SAFE_MORPH_FOR_DECOMPR(vectorlib::avx512<vectorlib::v512<uint64_t>>)
#endif
            
    #undef MAKE_SAFE_MORPH_FOR_DECOMPR

    // ------------------------------------------------------------------------
    // Format type_packing_f
    // ------------------------------------------------------------------------
    // Compression

    template<class t_layout> \
    struct safe_morph_t< type_packing_f< t_layout >, uncompr_f> { \
        using dst_f = type_packing_f<t_layout>; \
        using src_f = uncompr_f; \
         \
        static \
        const column<dst_f> * \
        apply(const column<src_f> * inCol) { \
            return morph<vectorlib::scalar<vectorlib::v64<uint64_t>>, dst_f, src_f>(inCol); \
        }; \
    };
    
    //Decompression

    template<class t_layout> \
    struct safe_morph_t<uncompr_f, type_packing_f<t_layout> > { \
        using dst_f = uncompr_f; \
        using src_f = type_packing_f<t_layout>; \
         \
        static \
        const column<dst_f> * \
        apply(const column<src_f> * inCol) { \
            return morph<vectorlib::scalar<vectorlib::v64<uint64_t>>, dst_f, src_f>(inCol); \
        }; \
    };

}

#endif //MORPHSTORE_CORE_MORPHING_SAFE_MORPH_H
