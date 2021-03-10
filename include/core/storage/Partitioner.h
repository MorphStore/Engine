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


#ifndef MORPHSTORE_PARTITIONER_H
#define MORPHSTORE_PARTITIONER_H

#include <stdlibs>
#include <storage>
#include <core/utils/helper.h>

namespace morphstore {

    /**
     * Base class for all partitioner classes
     */
    class Partitioner {
      public:
        enum class PartitioningType {
            Logical,
            Physical,
            None
        };
        
        static const Partitioner::PartitioningType partitioningType = Partitioner::PartitioningType::None;
        //
    };
//
//    /// to enable LogicalPartitioner<TStorage, TBase>
//    template<IStorage TStorage, IArithmetic TBase, uint64_t alignment = MSV_MEMORY_MANAGER_ALIGNMENT_BYTE>
//    class LogicalPartitioner : Partitioner {
//        //
//    };
//
//
//    template<IFormat TFormat, IArithmetic TBase, uint64_t alignment>
//    class LogicalPartitioner<column<TFormat>, TBase, alignment> : Partitioner {
//      public:
//        static const Partitioner::PartitioningType partitioningType = Partitioner::PartitioningType::Logical;
//
//        static
//        std::vector<column<TFormat>*>* apply(column<TFormat>* orig, uint64_t partitionCount){
//            /// number of values that fit in one aligned chunk
//            const uint64_t chunkSize = alignment / sizeof(TBase);
//
//            /// retrieve pointer to data array
//            TBase * baseData = orig->get_data();
//            /// elements
//            uint64_t dataSize = orig->get_count_values();
//
//            /// count of aligned data chunks
//            uint64_t chunks = dataSize / chunkSize + ((dataSize % chunkSize) ? 1 : 0);
//
//            /// elements per partition
//            uint64_t partitionSize = chunkSize * (chunks / partitionCount + ((chunks % partitionCount) ? 1 : 0));
//
//            auto partitionSet = new std::vector<column<TFormat>* >;
//            for(uint64_t i = 0; i < partitionCount; ++i){
//                uint64_t begin = i * partitionSize;
//                uint64_t end = begin + partitionSize;
//                if(end > dataSize)
//                    end = dataSize;
//
//                VirtualColumn<TFormat, TBase> * virtualColumn;
//                if(begin > end) {
//                    /// create empty virtual column
//                    virtualColumn = new VirtualColumn<TFormat, TBase>(
//                      nullptr,
//                      std::move(column_meta_data(0, 0, 0, 0)));
//                } else {
//                    /// create new virtual column
//                    virtualColumn = new VirtualColumn<TFormat, TBase>(
//                      (baseData + begin),
//                      std::move(column_meta_data(end - begin, sizeof(TBase) * (end - begin), 0, 0)));
//                }
//                /// add to set
//                partitionSet->push_back(virtualColumn);
//            }
//
//            return partitionSet;
//        };
//    };
    

    /// to enable LogicalPartitioner<TStorage, TBase>
//    template<IStorage TStorage, IArithmetic TBase, uint64_t alignment = MSV_MEMORY_MANAGER_ALIGNMENT_BYTE>
//    class LogicalPartitioner : Partitioner {

//    };
    
    
    //// @todo === New Implementation ===
    class LogicalPartitioner : Partitioner {
      public:
        static const Partitioner::PartitioningType partitioningType = Partitioner::PartitioningType::Logical;
        
        template<IArithmetic TBase, IFormat TFormat, uint64_t alignment = 64>
        static
        std::vector<column<TFormat>*>* apply(const column<TFormat>* orig, uint64_t partitionCount){
            /// number of values that fit in one aligned chunk
            const uint64_t chunkSize = alignment / sizeof(TBase);
            
            /// retrieve pointer to data array
            TBase * baseData = orig->get_data();
            /// elements
            uint64_t dataSize = orig->get_count_values();
            
            /// count of aligned data chunks
            uint64_t chunks = dataSize / chunkSize + ((dataSize % chunkSize) ? 1 : 0);
            
            /// elements per partition
            uint64_t partitionSize = chunkSize * (chunks / partitionCount + ((chunks % partitionCount) ? 1 : 0));
            
            auto partitionSet = new std::vector<column<TFormat>* >;
            for(uint64_t i = 0; i < partitionCount; ++i){
                uint64_t begin = i * partitionSize;
                uint64_t end = begin + partitionSize;
                if(end > dataSize)
                    end = dataSize;
                
                VirtualColumn<TFormat, TBase> * virtualColumn;
                if(begin > end) {
                    /// create empty virtual column
                    virtualColumn = new VirtualColumn<TFormat, TBase>(
                      nullptr,
                      std::move(column_meta_data(0, 0, 0, 0)));
                } else {
                    /// create new virtual column
                    virtualColumn = new VirtualColumn<TFormat, TBase>(
                      (baseData + begin),
                      std::move(column_meta_data(end - begin, sizeof(TBase) * (end - begin), 0, 0)));
                }
                /// add to set
                partitionSet->push_back(virtualColumn);
            }
            
            return partitionSet;
        };
    };

    /// to enable LogicalPartitioner<TStorage, TBase>
    template<IStorage TStorage, IArithmetic TBase, uint64_t alignment = MSV_MEMORY_MANAGER_ALIGNMENT_BYTE>
    class PhysicalPartitioner : Partitioner {
        //
    };
    
    
    template<IFormat TFormat, IArithmetic TBase, uint64_t alignment>
    class PhysicalPartitioner<column<TFormat>, TBase, alignment> : Partitioner {
      public:
        static const Partitioner::PartitioningType partitioningType = Partitioner::PartitioningType::Physical;
        
        static
        std::vector<column<TFormat>*>* apply(column<TFormat>* orig, uint64_t partitionCount){
            /// @todo
        };
        
        /// @todo invalidate for const column<...>*
        
        static
        column<uncompr_f> * consolidate(std::vector<column<uncompr_f>*>* partitionSet){
            /// @todo
        }
    };
    
//    /// type_str specialization for LogicalPartitioner
//    template< typename...Args >
//    struct type_str<LogicalPartitioner<Args...>> {
//        static string apply() {
//            return "LogicalPartitioner<" + type_str<Args...>::apply() + ">";
//        }
//    };
    
} // namespace
#endif //MORPHSTORE_PARTITIONER_H
