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
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <tuple>

#include <cstdint>

namespace morphstore {
    
    // Note that the format uncompr_f is currently defined in "format.h".
    // @todo Move it here.
    
    
    // ************************************************************************
    // Interfaces for accessing compressed data
    // ************************************************************************

    // ------------------------------------------------------------------------
    // Sequential read
    // ------------------------------------------------------------------------

    template<
            class t_vector_extension,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    struct decompress_and_process_batch<
            t_vector_extension,
            uncompr_f,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountIn8,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & p_State
        ) {
            const base_t * inBase = reinterpret_cast<const base_t *>(p_In8);
            const size_t countInBase = convert_size<uint8_t, base_t>(p_CountIn8);

            for(size_t i = 0; i < countInBase; i += vector_element_count::value)
                t_op_vector<t_ve, t_extra_args ...>::apply(
                        vector::load<
                                t_ve,
                                vector::iov::ALIGNED,
                                vector_base_t_granularity::value
                        >(inBase + i),
                        p_State
                );
            
            p_In8 += p_CountIn8;
        }
    };
    
    // ------------------------------------------------------------------------
    // Sequential write
    // ------------------------------------------------------------------------

    template<class t_vector_extension>
    class selective_write_iterator<t_vector_extension, uncompr_f> {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

        base_t * m_OutBase;
        const base_t * const m_InitOutBase;

    public:
        selective_write_iterator(uint8_t * p_Out) :
                m_OutBase(reinterpret_cast<base_t *>(p_Out)),
                m_InitOutBase(m_OutBase)
        {
            //
        }

        MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(
                vector_t p_Data, vector_mask_t p_Mask, uint8_t p_MaskPopCount
        ) {
            vector::compressstore<
                    t_ve,
                    vector::iov::UNALIGNED,
                    vector_base_t_granularity::value
            >(m_OutBase, p_Data, p_Mask);
            m_OutBase += p_MaskPopCount;
        }
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(
                vector_t p_Data, vector_mask_t p_Mask
        ) {
            write(
                    p_Data,
                    p_Mask,
                    vector::count_matches<t_vector_extension>::apply(p_Mask)
            );
        }

        std::tuple<size_t, bool, uint8_t *> done() {
            return std::make_tuple(
                    0, true, reinterpret_cast<uint8_t *>(m_OutBase)
            );
        }

        size_t get_count_values() const {
            return m_OutBase - m_InitOutBase;
        }
    };
    
}
#endif //MORPHSTORE_CORE_MORPHING_UNCOMPR_H
