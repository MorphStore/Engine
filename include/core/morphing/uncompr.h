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
#include <core/morphing/write_iterator.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <vector/vecprocessor/tsubasa/extension_tsubasa.h>


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
        
                
        template<typename T = t_ve, typename std::enable_if<!(T::is_scalable::value), T>::type* = nullptr >
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountInLog,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & p_State
        ) {
            const base_t * inBase = reinterpret_cast<const base_t *>(p_In8);

            for(size_t i = 0; i < p_CountInLog; i += vector_element_count::value)
                t_op_vector<t_ve, t_extra_args ...>::apply(
                        vectorlib::load<
                                t_ve,
                                vectorlib::iov::ALIGNED,
                                vector_base_t_granularity::value
                        >(inBase + i),
                        p_State
                );
            
            p_In8 = reinterpret_cast<const uint8_t *>(inBase + p_CountInLog);
        }

        /*Scalable
        */
        template<typename T = t_ve, typename std::enable_if<T::is_scalable::value, T>::type* = nullptr >
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountInLog,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & p_State
        ) {
            const base_t * inBase = reinterpret_cast<const base_t *>(p_In8);

            if(p_CountInLog < vector_element_count::value){
                t_op_vector<t_ve, t_extra_args ...>::apply(
                        vectorlib::load<
                                t_ve,
                                vectorlib::iov::ALIGNED,
                                vector_base_t_granularity::value
                        >(inBase, p_CountInLog),
                        p_State,
                        p_CountInLog
                );
            }
            else {
                for(size_t i = 0; i < p_CountInLog; i += vector_element_count::value)
                    t_op_vector<t_ve, t_extra_args ...>::apply(
                            vectorlib::load<
                                    t_ve,
                                    vectorlib::iov::ALIGNED,
                                    vector_base_t_granularity::value
                            >(inBase + i),
                            p_State
                    );
            }
            
            p_In8 = reinterpret_cast<const uint8_t *>(inBase + p_CountInLog);
        }
    };




    // ------------------------------------------------------------------------
    // Random read
    // ------------------------------------------------------------------------

    template<class t_vector_extension>
    class random_read_access<t_vector_extension, uncompr_f> {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        const base_t * const m_Data;
                
    public:
        // Alias to itself, in this case.
        using type = random_read_access<t_vector_extension, uncompr_f>;
        
        random_read_access(const base_t * p_Data) : m_Data(p_Data) {
            //
        }

        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        vector_t get(const vector_t & p_Positions) {
            return vectorlib::gather<
                    t_ve,
                    vector_base_t_granularity::value,
                    sizeof(base_t)
            >(m_Data, p_Positions);
        }
    };
    
    // ------------------------------------------------------------------------
    // Sequential write
    // ------------------------------------------------------------------------
    
    /**
     * @brief Partial template specialization of write_iterator_base for
     * uncompressed data. Does not use a buffer internally.
     */
    template<class t_vector_extension>
    class write_iterator_base<t_vector_extension, uncompr_f> {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
        
    protected:
        base_t * m_OutBase;
        
    private:
        const base_t * const m_InitOutBase;
        
    protected:
        write_iterator_base(uint8_t * p_Out) :
                m_OutBase(reinterpret_cast<base_t *>(p_Out)),
                m_InitOutBase(m_OutBase)
        {
            //
        }
        
    public:
        std::tuple<size_t, uint8_t *, uint8_t *> done() {
            return std::make_tuple(
                    0,
                    reinterpret_cast<uint8_t *>(m_OutBase),
                    reinterpret_cast<uint8_t *>(m_OutBase)
            );
        }

        size_t get_count_values() const {
            return m_OutBase - m_InitOutBase;
        }
    };
    
    /**
     * @brief Partial template specialization of selective_write_iterator for
     * uncompressed data. Does not use a buffer internally.
     */
    template<class t_vector_extension>
    class selective_write_iterator<t_vector_extension, uncompr_f> :
            public write_iterator_base<t_vector_extension, uncompr_f>
    {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)

    public:
        selective_write_iterator(uint8_t * p_Out) :
                write_iterator_base<t_vector_extension, uncompr_f>(p_Out)
        {
            //
        }

        MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(
                vector_t p_Data, vector_mask_t p_Mask, uint8_t p_MaskPopCount
        ) {
            vectorlib::compressstore<
                    t_vector_extension,
                    vectorlib::iov::UNALIGNED,
                    vector_base_t_granularity::value
            >(this->m_OutBase, p_Data, p_Mask);
            this->m_OutBase += p_MaskPopCount;
        }
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(
                vector_t p_Data, vector_mask_t p_Mask
        ) {
            write(
                    p_Data,
                    p_Mask,
                    vectorlib::count_matches<t_vector_extension>::apply(p_Mask)
            );
        }
    };
    
    /**
     * @brief Partial template specialization of nonselective_write_iterator
     * for uncompressed data. Does not use a buffer internally.
     */
    template<class t_vector_extension>
    class nonselective_write_iterator<t_vector_extension, uncompr_f> :
            public write_iterator_base<t_vector_extension, uncompr_f>
    {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)

    public:
        nonselective_write_iterator(uint8_t * p_Out) :
                write_iterator_base<t_vector_extension, uncompr_f>(p_Out)
        {
            //
        }

        MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(vector_t p_Data) {
            vectorlib::store<
                    t_vector_extension,
                    vectorlib::iov::ALIGNED,
                    vector_base_t_granularity::value
            >(this->m_OutBase, p_Data);
            this->m_OutBase += vector_element_count::value;
        }
    };
    
}
#endif //MORPHSTORE_CORE_MORPHING_UNCOMPR_H
