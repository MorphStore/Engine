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
 
#include <iostream>
#include <vector>

#include <core/virtual/partitioning.h>
#include <vector/vector_extension_structs.h>


using namespace vectorlib;
using namespace virtuallib;
using namespace std;


static bool test_logical_partitioning(){
    bool success = true;
    
    cout << "Testing logical partitioning." << endl;
    using base_t = uint64_t;
    
    uint64_t columnElementSize = 1003;
    uint64_t partitionCount = 5;
    
    base_t * column_data;
    PartitionSet<base_t>* partitions;
    
    column_data = new base_t[columnElementSize];
    partitions = logical_partitioning<base_t>::apply(column_data, 1000, 5);
    
    if(
      partitions->dataPtrSet.at(0) != (column_data +    0) or partitions->sizeSet.at(0) != 200 or
      partitions->dataPtrSet.at(1) != (column_data +  200) or partitions->sizeSet.at(1) != 200 or
      partitions->dataPtrSet.at(2) != (column_data +  400) or partitions->sizeSet.at(2) != 200 or
      partitions->dataPtrSet.at(3) != (column_data +  600) or partitions->sizeSet.at(3) != 200 or
      partitions->dataPtrSet.at(4) != (column_data +  800) or partitions->sizeSet.at(4) != 200
      ) success = false;
//    print_partitions(partitions);
    
    delete column_data;
    delete partitions;
    column_data = new base_t[columnElementSize];
    partitions = logical_partitioning<base_t>::apply(column_data, 733, 27);
    
    if(
      partitions->dataPtrSet.at( 0) != (column_data +   0 * 28) or partitions->sizeSet.at( 0) != 28 or
      partitions->dataPtrSet.at( 1) != (column_data +   1 * 28) or partitions->sizeSet.at( 1) != 28 or
      partitions->dataPtrSet.at( 2) != (column_data +   2 * 28) or partitions->sizeSet.at( 2) != 28 or
      partitions->dataPtrSet.at( 3) != (column_data +   3 * 28) or partitions->sizeSet.at( 3) != 28 or
      partitions->dataPtrSet.at( 4) != (column_data +   4 * 28) or partitions->sizeSet.at( 4) != 28 or
      partitions->dataPtrSet.at( 5) != (column_data +   5 * 28) or partitions->sizeSet.at( 5) != 28 or
      partitions->dataPtrSet.at( 6) != (column_data +   6 * 28) or partitions->sizeSet.at( 6) != 28 or
      partitions->dataPtrSet.at( 7) != (column_data +   7 * 28) or partitions->sizeSet.at( 7) != 28 or
      partitions->dataPtrSet.at( 8) != (column_data +   8 * 28) or partitions->sizeSet.at( 8) != 28 or
      partitions->dataPtrSet.at( 9) != (column_data +   9 * 28) or partitions->sizeSet.at( 9) != 28 or
      partitions->dataPtrSet.at(10) != (column_data +  10 * 28) or partitions->sizeSet.at(10) != 28 or
      partitions->dataPtrSet.at(11) != (column_data +  11 * 28) or partitions->sizeSet.at(11) != 28 or
      partitions->dataPtrSet.at(12) != (column_data +  12 * 28) or partitions->sizeSet.at(12) != 28 or
      partitions->dataPtrSet.at(13) != (column_data +  13 * 28) or partitions->sizeSet.at(13) != 28 or
      partitions->dataPtrSet.at(14) != (column_data +  14 * 28) or partitions->sizeSet.at(14) != 28 or
      partitions->dataPtrSet.at(15) != (column_data +  15 * 28) or partitions->sizeSet.at(15) != 28 or
      partitions->dataPtrSet.at(16) != (column_data +  16 * 28) or partitions->sizeSet.at(16) != 28 or
      partitions->dataPtrSet.at(17) != (column_data +  17 * 28) or partitions->sizeSet.at(17) != 28 or
      partitions->dataPtrSet.at(18) != (column_data +  18 * 28) or partitions->sizeSet.at(18) != 28 or
      partitions->dataPtrSet.at(19) != (column_data +  19 * 28) or partitions->sizeSet.at(19) != 28 or
      partitions->dataPtrSet.at(20) != (column_data +  20 * 28) or partitions->sizeSet.at(20) != 28 or
      partitions->dataPtrSet.at(21) != (column_data +  21 * 28) or partitions->sizeSet.at(21) != 28 or
      partitions->dataPtrSet.at(22) != (column_data +  22 * 28) or partitions->sizeSet.at(22) != 28 or
      partitions->dataPtrSet.at(23) != (column_data +  23 * 28) or partitions->sizeSet.at(23) != 28 or
      partitions->dataPtrSet.at(24) != (column_data +  24 * 28) or partitions->sizeSet.at(24) != 28 or
      partitions->dataPtrSet.at(25) != (column_data +  25 * 28) or partitions->sizeSet.at(25) != 28 or
      partitions->dataPtrSet.at(26) != (column_data +  26 * 28) or partitions->sizeSet.at(26) != 5
      ) success = false;
//    print_partitions(partitions);
    
    delete column_data;
    delete partitions;
    column_data = new base_t[columnElementSize];
    partitions = logical_partitioning<base_t>::apply(column_data, 23756, 3);
    
    if(
      partitions->dataPtrSet.at(0) != (column_data + 0 * 7919) or partitions->sizeSet.at(0) != 7919 or
      partitions->dataPtrSet.at(1) != (column_data + 1 * 7919) or partitions->sizeSet.at(1) != 7919 or
      partitions->dataPtrSet.at(2) != (column_data + 2 * 7919) or partitions->sizeSet.at(2) != 7918
      ) success = false;
//    print_partitions(partitions);
    delete column_data;
    delete partitions;
    
    return success;
}


int main() {
    bool success = true;
    if(!test_logical_partitioning()) {
        cout << "Error. Partitioning Object logical_partitioning does not work correct." << endl;
        success = false;
    }

    /// .. other tests

    if(success)
        cout << "All partitioning tests are successful." << endl;
    else
        cout << "There are some errors with partitioning testing." << endl;
        
    return !success;
}
