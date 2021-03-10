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


#ifndef MORPHSTORE_INCLUDE_CORE_STORAGE_VIRTUALCOLUMN_H
#define MORPHSTORE_INCLUDE_CORE_STORAGE_VIRTUALCOLUMN_H

#include <core/storage/column.h>
#include <bitset>

namespace morphstore {

    template<IFormat TFormat, IArithmetic base_t_>
    class VirtualColumn : public column<TFormat> {
      public:
        using base_t = base_t_;
        
        /// befriend Partitioned Column
        template<IFormat, IArithmetic, IPartitioner>
        friend class PartitionedColumn;
        
        /**
         * Constructor of a virtual column.
         * @tparam base_t data type
         * @param alignedData pointer to existing data
         * @param metaData meta data object for this column
         */
        VirtualColumn(base_t_ * alignedData, column_meta_data&& metaData) :
            column<TFormat>::column( voidptr_t(alignedData), std::move(metaData) )
        {
//            std::cout << "New Virtual Column" << std::endl;
//            std::cout << "Pointer   : " << std::bitset<64>(reinterpret_cast<uint64_t>(alignedData)) << " = " << reinterpret_cast<uint64_t>(alignedData) << std::endl;
//            std::cout << "Alignment : " << std::bitset<64>(MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE) << " = " << MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE << std::endl;
//            std::cout << "Difference: " << std::bitset<64>((reinterpret_cast<uint64_t>(alignedData) & MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE)) << " = " << (reinterpret_cast<uint64_t>(alignedData) & MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE) << std::endl;
            /// check if data array is correctly aligned
            if(reinterpret_cast<uint64_t>(alignedData) & MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE) {
                std::stringstream s;
                s << "Column Constructor: given data pointer is not " << MSV_MEMORY_MANAGER_ALIGNMENT_BYTE << " Byte aligned.";
                throw std::runtime_error(s.str());
            }
        }
        
        ~VirtualColumn() = default;
    };
    
} // namespace

#endif //MORPHSTORE_INCLUDE_CORE_STORAGE_VIRTUALCOLUMN_H
