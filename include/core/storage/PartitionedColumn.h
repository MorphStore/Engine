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


#ifndef MORPHSTORE_PARTITIONEDCOLUMN_H
#define MORPHSTORE_PARTITIONEDCOLUMN_H

#include <core/storage/column_helper.h>
#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/morph_batch.h>
#include <core/utils/basic_types.h>
#include <core/utils/helper_types.h>
#include <core/storage/column.h>
#include <core/storage/Partitioner.h>

#include <type_traits>
#include <forward>


namespace morphstore {
//
//    template<IFormat TFormat, IArithmetic TBase, IPartitioner TPartitioner>
//    class PartitionedColumn : public Storage {
//        static_assert(
//          always_false<TPartitioner>::value,
//          "Column Format and Base Type of the PartitionedColumn and the given Partitioner must be the same!");
//    };
//
//    template<IFormat TFormat, IArithmetic TBase, template<typename, typename, uint64_t> class TPartitioner>
//    class PartitionedColumn<TFormat, TBase, TPartitioner<column<TFormat>, TBase, 64>> : public Storage {
//        using TPartitioner_ = TPartitioner<column<TFormat>, TBase, 64>;
//      private:
//        std::vector<column<TFormat>*>* partitionSet;
//        void * dataUnaligned;
//        voidptr_t data;
//
//        static const Partitioner::PartitioningType partitioningType = TPartitioner_::partitioningType;
//
//      public:
//
//        PartitionedColumn(column<TFormat> * prey, const uint64_t count) : data(prey->m_Data) {
//            partitionSet = TPartitioner_::apply(prey, count);
//            /// take over data ptr (@todo only for logical partitioning, move into partitioner?)
//            dataUnaligned = prey->m_DataUnaligned;
//
//            /// set column to virtual to avoid deletion of data
//            prey->is_virtual = true;
//            /// destroy column object
//            delete prey;
//        }
//
//        PartitionedColumn() : data(nullptr), dataUnaligned(nullptr) {
//            partitionSet = new std::vector<column<TFormat>*>();
//        }
//
//        column<TFormat>* operator[](size_t index){
//            return (*partitionSet)[index];
//        }
//
//        void repartition(uint64_t partitionCount){
//            //?
//        }
//
//        void addPartition(column<TFormat>* newPartition){
//            if constexpr(partitioningType == Partitioner::PartitioningType::Logical){
//                throw std::runtime_error("It is prohibited to add partitions to a logical partitioned column!");
//            } else {
//                partitionSet->push_back(newPartition);
//            }
//        }
//
//        uint64_t getPartitionCount(){
//            return partitionSet->size();
//        }
//    };

    //// @todo === New Implementation ===
    template<IPartitioner TPartitioner, IFormat TFormat, IArithmetic TBase = uint64_t>
    class PartitionedColumn : public Storage {
      private:
        std::vector<column<TFormat>*>* partitionSet;
        void * dataUnaligned;
        voidptr_t data;
        
        static const Partitioner::PartitioningType partitioningType = TPartitioner::partitioningType;
        
      public:
        
        PartitionedColumn(column<TFormat> * prey, const uint64_t count) : data(prey->m_Data) {
            partitionSet = TPartitioner::template apply<TBase>(prey, count);
            /// take over data ptr (@todo only for logical partitioning, move into partitioner?)
            dataUnaligned = prey->m_DataUnaligned;
            
            /// set column to virtual to avoid deletion of data
            prey->is_virtual = true;
            /// destroy column object
            delete prey;
        }
        
//        PartitionedColumn(column<TFormat> const * noPrey, const uint64_t count) : data(noPrey->m_Data) {
//            partitionSet = TPartitioner::template apply<TBase>(noPrey, count);
//            /// take over data ptr (@todo only for logical partitioning, move into partitioner?)
//            dataUnaligned = noPrey->m_DataUnaligned;
//        }
        
        PartitionedColumn() : data(nullptr), dataUnaligned(nullptr) {
            partitionSet = new std::vector<column<TFormat>*>();
        }
        
        const column<TFormat>* operator[](size_t index){
            return (*partitionSet)[index];
        }

        void repartition(uint64_t partitionCount){
            //?
        }
        
        void addPartition(column<TFormat>* newPartition){
            if constexpr(partitioningType == Partitioner::PartitioningType::Logical){
                throw std::runtime_error("It is prohibited to add partitions to a logical partitioned column!");
            } else {
                partitionSet->push_back(newPartition);
            }
        }
        
        uint64_t getPartitionCount(){
            return partitionSet->size();
        }
    };
    
//    template<typename TPartitioner, template<typename> class TStorage, typename TFormat>
//    PartitionedColumn<TPartitioner>::PartitionedColumn(TStorage<TFormat>*, const uint64_t) -> PartitionedColumn<TPartitioner, TFormat>;
    
    
    
    /// type_str specialization for PartitionedColumn
    template< typename...Args >
    struct type_str<PartitionedColumn<Args...>> {
        static string apply() {
            return "PartitionedColumn<" + type_str<Args...>::apply() + ">";
        }
    };

}


#endif //MORPHSTORE_PARTITIONEDCOLUMN_H
