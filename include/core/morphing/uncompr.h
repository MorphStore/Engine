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
 * @file uncompr.h
 * @brief Facilities for accessing uncompressed data via the same interface as
 * compressed data.
 */

#ifndef MORPHSTORE_CORE_MORPHING_UNCOMPR_H
#define MORPHSTORE_CORE_MORPHING_UNCOMPR_H

#include <core/morphing/format.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/general_vector.h>
#include <vector/primitives/compare.h>
#include <vector/primitives/io.h>
// @todo The following includes should not be necessary.
#include <vector/scalar/extension_scalar.h>
#include <vector/scalar/primitives/compare_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>
#include <vector/simd/sse/primitives/compare_sse.h>
#include <vector/simd/sse/primitives/compare_sse.h>

#include <cstdint>

namespace morphstore {

#if 0
    // Hand-written scalar.
    
    template<template<class> class t_op_processing_unit>
    struct decompress_and_process_batch<vector::scalar<vector::v64<uint64_t>>, uncompr_f, t_op_processing_unit> {
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountIn8,
                typename t_op_processing_unit<vector::scalar<vector::v64<uint64_t>>>::state_t & p_State
        ) {
            const uint64_t * in64 = reinterpret_cast<const uint64_t *>(p_In8);
            const size_t countIn64 = convert_size<uint8_t, uint64_t>(p_CountIn8);

            for(size_t i = 0; i < countIn64; i++)
                t_op_processing_unit<vector::scalar<vector::v64<uint64_t>>>::apply(in64[i], p_State);
        }
    };
    
#else
    // Generic with vector-lib.
    
    template<class t_vector_extension, template<class> class t_op_processing_unit>
    struct decompress_and_process_batch<t_vector_extension, uncompr_f, t_op_processing_unit> {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountIn8,
                typename t_op_processing_unit<t_ve>::state_t & p_State
        ) {
            const base_t * inBase = reinterpret_cast<const base_t *>(p_In8);
            const size_t countInBase = convert_size<uint8_t, base_t>(p_CountIn8);

            for(size_t i = 0; i < countInBase; i += vector_element_count::value)
                t_op_processing_unit<t_ve>::apply(
                        vector::load<
                                t_ve,
                                vector::iov::ALIGNED,
                                vector_base_t_granularity::value
                        >(inBase + i),
                        p_State
                );
        }
    };
#endif

#if 0
    // Hand-written scalar.
    
    template<>
    class write_iterator<vector::scalar<vector::v64<uint64_t>>, uncompr_f> {
        IMPORT_VECTOR_BOILER_PLATE(vector::scalar<vector::v64<uint64_t>>)

        uint64_t * m_Out64;
        const uint64_t * const m_InitOut64;

    public:
        write_iterator(uint8_t * p_Out) :
                m_Out64(reinterpret_cast<uint64_t *>(p_Out)),
                m_InitOut64(m_Out64)
        {
            //
        }

        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        void write(vector_t p_Data, vector_mask_t p_Mask) {
            vector::compressstore<
                    vector::scalar<vector::v64<uint64_t>>,
                    vector::iov::UNALIGNED, 
                    vector_base_t_granularity::value
            >(m_Out64, p_Data, p_Mask);
            // @todo Use primitive.
            m_Out64 += p_Mask;
        }

        void done() {
            //
        }

        size_t get_count() const {
            return m_Out64 - m_InitOut64;
        }
    };
#else
    // Generic with vector-lib.
    
    template<class t_vector_extension>
    class write_iterator<t_vector_extension, uncompr_f> {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

        base_t * m_OutBase;
        const base_t * const m_InitOutBase;

    public:
        write_iterator(uint8_t * p_Out) :
                m_OutBase(reinterpret_cast<base_t *>(p_Out)),
                m_InitOutBase(m_OutBase)
        {
            //
        }

        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        void write(vector_t p_Data, vector_mask_t p_Mask) {
            vector::compressstore<
                    t_ve,
                    vector::iov::UNALIGNED,
                    vector_base_t_granularity::value
            >(m_OutBase, p_Data, p_Mask);
            m_OutBase += vector::count_matches<t_ve>::apply(p_Mask);
        }

        void done() {
            //
        }

        size_t get_count() const {
            return m_OutBase - m_InitOutBase;
        }
    };
#endif
    
}
#endif //MORPHSTORE_CORE_MORPHING_UNCOMPR_H
