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
 * @file data_properties.h
 * @brief A utility for determining some important data characteristics of a
 * column.
 * 
 * Note that this code is not tailored for efficiency.
 */

#ifndef MORPHSTORE_CORE_UTILS_DATA_PROPERTIES_H
#define MORPHSTORE_CORE_UTILS_DATA_PROPERTIES_H

#include <core/morphing/format.h> // for uncompr_f
#include <core/storage/column.h>
#include <core/utils/math.h>

#include <limits>
#include <stdexcept>
#include <unordered_set>

#include <cstdint>

namespace morphstore {
    
    /**
     * @brief A container for important data characteristics of a column.
     */
    class data_properties {
        uint64_t m_Min;
        uint64_t m_Max;
        size_t m_BwHist[std::numeric_limits<uint64_t>::digits];
        bool m_IsSortedAsc;
        size_t m_DistinctCount;
        bool m_IsUnique;
        
    public:
        /**
         * @brief Creates a record of data properties by analyzing the given
         * column.
         * @param p_Col The column to analyze.
         * @param p_IsKnownUnique Whether the column is known to be unique. If
         * `true` is specified, then the number of distinct values will not be
         * determined explicitly, which saves time.
         */
        data_properties(
                const column<uncompr_f> * p_Col,
                bool p_IsKnownUnique = false
        ) {
            const size_t count = p_Col->get_count_values();
            const uint64_t * data = p_Col->get_data();
            
            m_Min = std::numeric_limits<uint64_t>::max();
            m_Max = 0;
            for(
                    unsigned bw = 0;
                    bw < std::numeric_limits<uint64_t>::digits;
                    bw++
            )
                m_BwHist[bw] = 0;
            m_IsSortedAsc = true;
            
            uint64_t prevVal = data[0];
            std::unordered_set<uint64_t> distinctValues;
            
            for(size_t i = 0; i < count; i++) {
                const uint64_t val = data[i];
                
                if(val < m_Min) m_Min = val;
                if(val > m_Max) m_Max = val;
                m_BwHist[effective_bitwidth(val) - 1]++;
                m_IsSortedAsc = m_IsSortedAsc && val >= prevVal;
                
                prevVal = val;
                if(!p_IsKnownUnique)
                    distinctValues.emplace(val);
            }
            
            if(p_IsKnownUnique) {
                m_DistinctCount = count;
                m_IsUnique = true;
            }
            else {
                m_DistinctCount = distinctValues.size();
                m_IsUnique = count == m_DistinctCount;
            }
        }
        
        /**
         * @brief Returns the smallest value in the column.
         * @return The smallest value in the column.
         */
        uint64_t get_min() const {
            return m_Min;
        }
        
        /**
         * @brief Returns the greatest value in the column.
         * @return The greatest value in the column.
         */
        uint64_t get_max() const {
            return m_Max;
        }
        
        /**
         * @brief Returns the absolute frequency of values with the specified
         * number of effective bits.
         * @param p_Bw The bit width in [1, 64].
         * @return The absolute frequency of values with the specified number
         * of effective bits.
         */
        size_t get_bw_hist(unsigned p_Bw) const {
            if(p_Bw < 1 || p_Bw > std::numeric_limits<uint64_t>::max())
                throw std::runtime_error("the bit width must be in [1, 64");
            return m_BwHist[p_Bw - 1];
        }
        
        /**
         * @brief Returns whether the data is sorted in ascending order.
         * @return `true` if the data is sorted, `false` otherwise.
         */
        bool is_sorted_asc() const {
            return m_IsSortedAsc;
        }
        
        /**
         * @brief Returns the number of distinct data elements.
         * @return The number of distinct data elements.
         */
        size_t get_distinct_count() const {
            return m_DistinctCount;
        }
        
        /**
         * @brief Returns whether the data is unique, i.e., whether each value
         * occurs only once.
         * @return `true` if the data is unique, `false` otherwise.
         */
        bool is_unique() const {
            return m_IsUnique;
        }
    };
}
#endif //MORPHSTORE_CORE_UTILS_DATA_PROPERTIES_H
