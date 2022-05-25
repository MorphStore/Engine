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


#ifndef MORPHSTORE_PARTITIONING_H
#define MORPHSTORE_PARTITIONING_H

#include <vector/vv/extension_virtual_vector.h>
#include <core/storage/column.h>

//using namespace morphstore;

namespace virtuallib {
    
    template<typename base_t>
    struct Partition {
        Partition(base_t * dataPtr, uint64_t size) : dataPtr(dataPtr), size(size) {};
        base_t * dataPtr;
        uint64_t size;
    };
    
    template<typename base_t>
    struct PartitionSet {
        std::vector<base_t *> dataPtrSet;
        std::vector<uint64_t> sizeSet;
        
        void add(base_t * dataPtr, uint64_t size){
            dataPtrSet.push_back(dataPtr);
            sizeSet.push_back(size);
        }
    };
    
    
    template<typename base_t>
    static void print_partitions(std::vector<Partition<base_t>> * vec){
        uint64_t partIdx = 0;
        base_t * begin = vec->at(0).dataPtr;
        for(auto partition : *vec){
            std::cout << "Partition " << partIdx++ << " begins at " << partition.dataPtr << " (+"
              << (uint64_t) (partition.dataPtr - begin) << ") and has "
              << partition.size << " elements." << std::endl;
        }
    }
    
    template<typename base_t>
    static void print_partitions(PartitionSet<base_t> * set){
        base_t * begin = set->dataPtrSet.at(0);
        for(uint64_t i = 0; i < set->dataPtrSet.size(); ++i){
            std::cout << "Partition " << i << " begins at " << set->dataPtrSet.at(i) << " (+"
              << (uint64_t) (set->dataPtrSet.at(i) - begin) << ") and has "
              << set->sizeSet.at(i) << " elements." << std::endl;
        }
    }

    
    /// @todo @eric Add element pack option (i.e. a specifier how many elements shall go to the same partition to avoid scalar remain when vectorizing.
    template<typename base_t>
    struct logical_partitioning {
//        static
//        std::vector<Partition<base_t>>* apply(base_t * inColumnDataPtr, uint64_t elementCount, uint64_t partitionCount){
//            uint64_t valuesPerPartition = elementCount / partitionCount
//                                          + (elementCount % partitionCount ? 1 : 0);
//
//            /// Ptr behind last element
//            base_t * endPtr = inColumnDataPtr + elementCount;
//
//            auto * result = new std::vector<Partition<base_t>>;
//            for(uint64_t i = 0; i < partitionCount; ++i){
//                base_t * begin = inColumnDataPtr + i * valuesPerPartition;
//                base_t * end = begin + valuesPerPartition;
//                if(end > endPtr) end = endPtr;
//                uint64_t size = end - begin;
//                result->emplace_back(begin, size);
////                result->push_back(in_column_data + i * valuesPerPartition);
//            }
//            return result;
//        }

        static
        PartitionSet<base_t> * apply(base_t * inColumnDataPtr, uint64_t elementCount, uint64_t partitionCount){
            uint64_t valuesPerPartition = elementCount / partitionCount
                                          + (elementCount % partitionCount ? 1 : 0);
            
            /// Ptr behind last element
            base_t * endPtr = inColumnDataPtr + elementCount;
            
            auto * result = new PartitionSet<base_t>;
            for(uint64_t i = 0; i < partitionCount; ++i){
                base_t * begin = inColumnDataPtr + i * valuesPerPartition;
                base_t * end = begin + valuesPerPartition;
                if(end > endPtr) end = endPtr;
                uint64_t size = end - begin;
                result->add(begin, size);
//                result->push_back(in_column_data + i * valuesPerPartition);
            }
            return result;
        }
    };
}


#endif //MORPHSTORE_PARTITIONING_H
