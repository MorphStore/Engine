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
 * @file replicated_write_iterator.h
 * @brief Interfaces and default implementations for writing data to the replicated
 * column in any format.
 */

#ifndef MORPHSTORE_CORE_STORAGE_REPLICATED_WRITE_ITERATOR_H
#define MORPHSTORE_CORE_STORAGE_REPLICATED_WRITE_ITERATOR_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/morph.h>
#include <core/morphing/default_formats.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <tuple>
#include <cstdint>
#include <cstring>
#include "replicated_column.h"

namespace morphstore {

    /**
     * @brief General default implementation of replicated_write_iterator_base. Uses a
     * buffer internally.
     */
    template<class t_vector_extension>
    class replicated_write_iterator_base {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)

        static const size_t m_CountBuffer = 2048;

        MSV_CXX_ATTRIBUTE_ALIGNED(vector_size_byte::value) base_t m_StartBuffer[
                m_CountBuffer + vector_element_count::value - 1
        ];

        size_t m_Count;

    protected:
        replicated_column * rc;
        size_t countRep = 0;
        std::vector<uint8_t *> m_OutReplicated = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        std::vector<uint8_t *> m_InitOutReplicated = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

        base_t * m_Buffer;
        base_t * const m_EndBuffer;

        void compress_buffer() {
            for (size_t i = 0; i < countRep; i++)
            {

            const uint8_t * buffer8 = reinterpret_cast<uint8_t *>(
                    m_StartBuffer
            );

            switch (rc->compressed->m_ReplicatedMetaData.m_Replicas[i].second.format)
            {
            case (0): // UNCOMPR
            morph_batch<t_ve, uncompr_f, uncompr_f>(
                    buffer8, m_OutReplicated[i], m_CountBuffer
            );
            break;
            case (1): // STATICBP, bitwidth = 32
            morph_batch<t_ve, DEFAULT_STATIC_VBP_F(t_ve, 32), uncompr_f>(
                    buffer8, m_OutReplicated[i], m_CountBuffer
            );
            break;
            case (2): // DYNAMICBP
            morph_batch<t_ve, DEFAULT_DYNAMIC_VBP_F(t_ve), uncompr_f>(
                    buffer8, m_OutReplicated[i], m_CountBuffer
            );
            break;
            }
            }

            size_t overflow = m_Buffer - m_EndBuffer;
            memcpy(m_StartBuffer, m_EndBuffer, overflow * sizeof(base_t));
            m_Buffer = m_StartBuffer + overflow;
            m_Count += m_CountBuffer;
            }

        replicated_write_iterator_base() :
                m_Count(0),
                m_Buffer(m_StartBuffer),
                m_EndBuffer(m_StartBuffer + m_CountBuffer)
        {
            //
        }

    public:
        /**
         * @brief Makes sure that all possibly buffered data is stored to the
         * output. This function should always be called after the last call to
         * `write`.
         *
         * @return void
         */
         void done() {
            // Lock the replicated column for metadata update
            pthread_rwlock_wrlock(rc->columnLock);

            const size_t countLog = m_Buffer - m_StartBuffer;
            std::vector<size_t> outSizeComprByte(countRep);
            if(countLog) {
                const size_t outCountLogCompr = round_down_to_multiple(
                        countLog, DEFAULT_STATIC_VBP_F(t_ve, 32)::m_BlockSize
                );
                for (size_t i = 0; i < countRep; i++)
            {
            rc->uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data = 0;
            const uint8_t * buffer8 = reinterpret_cast<uint8_t *>(
                    m_StartBuffer
            );

            switch (rc->compressed->m_ReplicatedMetaData.m_Replicas[i].second.format)
            {
            case (0): // UNCOMPR
            morph_batch<t_ve, uncompr_f, uncompr_f>(
                    buffer8, m_OutReplicated[i], outCountLogCompr
            );
            break;
            case (1): // STATICBP, bitwidth = 32
            morph_batch<t_ve, DEFAULT_STATIC_VBP_F(t_ve, 32), uncompr_f>(
                    buffer8, m_OutReplicated[i], outCountLogCompr
            );
            break;
            case (2): // DYNAMICBP
            morph_batch<t_ve, DEFAULT_DYNAMIC_VBP_F(t_ve), uncompr_f>(
                    buffer8, m_OutReplicated[i], outCountLogCompr
            );
            break;
            }

                outSizeComprByte[i] = m_OutReplicated[i] - m_InitOutReplicated[i];

                const size_t outCountLogRest = countLog - outCountLogCompr;
                if(outCountLogRest) {
                    const size_t sizeOutLogRest = uncompr_f::get_size_max_byte(outCountLogRest);

                    memcpy(
                            reinterpret_cast<uint8_t *>(rc->uncompressed->m_ReplicatedMetaData.m_Replicas[i].first) +
                               this->rc->uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data,
                            m_StartBuffer + outCountLogCompr,
                            sizeOutLogRest
                    );
                    // Update size of uncompressed buffer
                    rc->uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data += sizeOutLogRest;
                    // Ensure persistency for uncompressed NVRAM-resident parts
                    if (rc->uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM)
                        nvram_flush(rc->uncompressed->m_ReplicatedMetaData.m_Replicas[i].first, this->rc->uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data);
                }
                m_Count += countLog;
                }

            }

            for (size_t i = 0; i < countRep; i++)
            {
                // Update size of compressed buffer
                rc->compressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data += m_OutReplicated[i] - m_InitOutReplicated[i];
                // Ensure persistency for compressed NVRAM-resident parts
                if (rc->compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM)
                    nvram_flush(m_InitOutReplicated[i], this->rc->uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data);
            }

            pthread_rwlock_unlock(rc->columnLock);

            return;
        }

        /**
         * @brief Returns the number of logical data elements that were stored
         * using this instance.
         * @return The number of logical data elements stored using this
         * instance.
         */
        size_t get_count_values () const {
            return m_Count;
        }
    };

    /**
     * @brief The interface for writing compressed data non-selectively.
     *
     * The default implementation buffers the appended data elements in
     * uncompressed form first. When the internal buffer is full, then the
     * batch-level morph-operator is used to compress the buffer and append the
     * compresed data to the output column's data buffer. **This interface does
     * not need to be implemented for new formats**, since the default
     * implementation can always be used as long as there is a specialization
     * of the batch-level morph-operator for the respective output format.
     */
    template<class t_vector_extension>
    class nonselective_replicated_write_iterator :
            public replicated_write_iterator_base<t_vector_extension>
    {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)

    public:

        nonselective_replicated_write_iterator(replicated_column* p_rc) :
                replicated_write_iterator_base<t_vector_extension>()
        {
            this->rc = p_rc;
            this->countRep = this->rc->compressed->m_ReplicatedMetaData.m_ReplicaCount;
            for (size_t i = 0; i < this->countRep; i++)
            {
               this->m_OutReplicated[i] = create_aligned_ptr(reinterpret_cast<uint8_t *>(this->rc->compressed->m_ReplicatedMetaData.m_Replicas[i].first))
                                        + this->rc->compressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data;
               this->m_InitOutReplicated[i] = create_aligned_ptr(reinterpret_cast<uint8_t *>(this->rc->compressed->m_ReplicatedMetaData.m_Replicas[i].first))
                                            + this->rc->compressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data;
            }

            // Check for uncompressed rest part
            auto pos = create_aligned_ptr(reinterpret_cast<uint64_t *>(this->rc->uncompressed->m_ReplicatedMetaData.m_Replicas[0].first));
            for (size_t i = 0; i < this->rc->uncompressed->m_ReplicatedMetaData.m_Replicas[0].second.size_data / sizeof(uint64_t); i++)
            {
               this->write(*pos++);
            }
        };

        /**
         * @brief Stores the given data vector to the output.
         * 
         * Internally, buffering may take place, such that it is not guaranteed
         * that the data is stored to the output immediately.
         * 
         * `done` must be called after the last call to this function to
         * guarantee that the data is stores to the output in any case.
         * 
         * @param p_Data
         */
        MSV_CXX_ATTRIBUTE_FORCE_INLINE void write(uint64_t p_Data) {

            *(this->m_Buffer++) = p_Data;
            if(MSV_CXX_ATTRIBUTE_UNLIKELY(this->m_Buffer >= this->m_EndBuffer))
                this->compress_buffer();
        }
};

}
#endif //MORPHSTORE_CORE_STORAGE_REPLICATED_WRITE_ITERATOR_H